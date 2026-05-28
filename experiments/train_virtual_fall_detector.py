from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.check_data import (
    compute_client_baselines,
    evaluate_predictions,
    extract_window_features,
    fit_weighted_logistic_regression,
    make_dataset,
    make_unseen_facility_dataset,
    predict_logistic,
    _lidar_frame_features,
)


MODEL_PATH = Path("results/virtual_fall_detector_5frame.npz")
USE_CLIENT_BASELINE = True


def metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    false_alarm = fp / max(1, fp + tn)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "false_alarm": false_alarm,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y_true, probs, min_recall=0.90):
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (probs >= threshold).astype(np.int64)
        m = metrics(y_true, y_pred)
        if m["recall"] < min_recall:
            continue
        candidate = (
            m["false_alarm"],
            -m["precision"],
            -m["accuracy"],
            threshold,
            m,
        )
        if best is None or candidate < best:
            best = candidate

    if best is not None:
        return float(best[3]), best[4]

    best_recall = None
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (probs >= threshold).astype(np.int64)
        m = metrics(y_true, y_pred)
        candidate = (-m["recall"], m["false_alarm"], threshold, m)
        if best_recall is None or candidate < best_recall:
            best_recall = candidate
    return float(best_recall[2]), best_recall[3]


def report_dataset(name, y):
    print(f"{name}: samples={len(y)} falls={int(y.sum())} fall_ratio={float(y.mean()):.4f}")


def window_features_for_bundles(bundles, baseline=None, window_size=5):
    center = window_size // 2
    features = []
    center_indices = []
    center_labels = []

    for start in range(0, len(bundles) - window_size + 1):
        window = bundles[start:start + window_size]
        per_frame = [_lidar_frame_features(b) for b in window]
        if any(f is None for f in per_frame):
            continue

        F = np.vstack(per_frame)
        z_mean = F[:, 0]
        z_std = F[:, 1]
        floor02 = F[:, 4]
        floor04 = F[:, 5]
        point_count = F[:, 6]
        xy_spread = F[:, 9]
        pressure = np.array([
            float(b.pressure.sum()) if b.pressure is not None else 0.0
            for b in window
        ], dtype=np.float64)
        center_with_pressure = np.r_[F[center], pressure[center]]

        diffs = np.diff(z_mean)
        pressure_diffs = np.diff(pressure)
        temporal_features = np.array([
            float(z_mean[0]),
            float(z_mean[-1]),
            float(np.min(z_mean)),
            float(np.max(z_mean)),
            float(np.max(z_mean) - np.min(z_mean)),
            float(z_mean[-1] - z_mean[0]),
            float(np.min(diffs)),
            float(np.max(diffs)),
            float(np.mean(np.abs(diffs))),
            float(np.max(z_std) - np.min(z_std)),
            float(floor02[-1] - floor02[0]),
            float(np.max(floor02)),
            float(floor04[-1] - floor04[0]),
            float(np.max(floor04)),
            float(np.min(point_count)),
            float(point_count[-1] - point_count[0]),
            float(np.max(xy_spread) - np.min(xy_spread)),
            float(pressure[center]),
            float(pressure[0]),
            float(pressure[-1]),
            float(np.max(pressure) - np.min(pressure)),
            float(pressure[-1] - pressure[0]),
            float(np.min(pressure_diffs)),
            float(np.max(pressure_diffs)),
        ], dtype=np.float64)
        feature_parts = [F[center], temporal_features]
        if baseline is not None:
            feature_parts.append(center_with_pressure - baseline)
        features.append(np.concatenate(feature_parts))
        center_bundle = window[center]
        center_indices.append(start + center)
        center_labels.append(1 if center_bundle.semantic_state == "ABNORMAL" else 0)

    return (
        np.array(features, dtype=np.float64),
        np.array(center_indices, dtype=np.int64),
        np.array(center_labels, dtype=np.int64),
    )


def alert_indices_from_probs(indices, probs, threshold, consecutive=2, cooldown=12):
    alerts = []
    streak = 0
    last_alert = -10**9

    for idx, prob in zip(indices, probs):
        if prob >= threshold:
            streak += 1
        else:
            streak = 0

        if streak >= consecutive and idx - last_alert >= cooldown:
            alerts.append(int(idx))
            last_alert = int(idx)
            streak = 0

    return alerts


def fall_episodes(labels):
    episodes = []
    start = None
    for i, label in enumerate(labels):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(labels) - 1))
    return episodes


def evaluate_alerts(data, model, threshold, title, tolerance=3, verbose=True):
    sequences = alert_sequences(data, model)
    return evaluate_alert_sequences(sequences, threshold, title, tolerance, verbose)


def alert_sequences(data, model):
    sequences = []
    baselines = compute_client_baselines(data) if USE_CLIENT_BASELINE else {}
    for client_id, bundles in data.items():
        X, indices, labels = window_features_for_bundles(
            bundles,
            baseline=baselines.get(client_id),
        )
        if len(X) == 0:
            continue
        _, probs = predict_logistic(X, model, threshold=0.5)
        sequences.append((indices, labels, probs))
    return sequences


def evaluate_alert_sequences(sequences, threshold, title, tolerance=3, verbose=True):
    total_episodes = 0
    detected_episodes = 0
    total_alerts = 0
    false_alerts = 0

    for indices, labels, probs in sequences:
        alerts = alert_indices_from_probs(indices, probs, threshold)
        episodes = fall_episodes(labels)

        total_episodes += len(episodes)
        total_alerts += len(alerts)

        matched_alerts = set()
        for ep_start, ep_end in episodes:
            hit = False
            for alert_i, alert in enumerate(alerts):
                if ep_start - tolerance <= alert <= ep_end + tolerance:
                    hit = True
                    matched_alerts.add(alert_i)
            if hit:
                detected_episodes += 1

        false_alerts += len(alerts) - len(matched_alerts)

    recall = detected_episodes / max(1, total_episodes)
    precision = (total_alerts - false_alerts) / max(1, total_alerts)

    if verbose:
        print(f"\n--- {title} event-level alerts ---")
        print(f"fall episodes:        {total_episodes}")
        print(f"detected episodes:    {detected_episodes}")
        print(f"alerts:               {total_alerts}")
        print(f"false alerts:         {false_alerts}")
        print(f"event recall:         {recall:.3f}")
        print(f"alert precision:      {precision:.3f}")
    return {
        "event_recall": recall,
        "alert_precision": precision,
        "fall_episodes": total_episodes,
        "alerts": total_alerts,
        "false_alerts": false_alerts,
    }


def choose_alert_threshold(data, model, min_event_recall=0.90):
    sequences = alert_sequences(data, model)
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        m = evaluate_alert_sequences(
            sequences,
            threshold,
            "threshold search",
            verbose=False,
        )
        if m["event_recall"] < min_event_recall:
            continue
        candidate = (
            m["false_alerts"],
            -m["alert_precision"],
            -m["event_recall"],
            threshold,
            m,
        )
        if best is None or candidate < best:
            best = candidate

    if best is not None:
        return float(best[3]), best[4]

    best_recall = None
    for threshold in np.linspace(0.05, 0.95, 181):
        m = evaluate_alert_sequences(
            sequences,
            threshold,
            "threshold search",
            verbose=False,
        )
        candidate = (-m["event_recall"], m["false_alerts"], threshold, m)
        if best_recall is None or candidate < best_recall:
            best_recall = candidate
    return float(best_recall[2]), best_recall[3]


def main():
    print("Building virtual fall detector datasets...")
    train_data = make_dataset(n_steps=12000, seed=123, abnormal_rate=0.08)
    validation_data = make_unseen_facility_dataset(n_steps=8000, seed=7)
    test_data = make_unseen_facility_dataset(n_steps=10000, seed=42)

    X_train, y_train = extract_window_features(
        train_data,
        use_client_baseline=USE_CLIENT_BASELINE,
    )
    X_val, y_val = extract_window_features(
        validation_data,
        use_client_baseline=USE_CLIENT_BASELINE,
    )
    X_test, y_test = extract_window_features(
        test_data,
        use_client_baseline=USE_CLIENT_BASELINE,
    )

    report_dataset("train fall-enriched", y_train)
    report_dataset("validation unseen", y_val)
    report_dataset("test unseen", y_test)

    print("\nTraining 5-frame weighted logistic regression...")
    model = fit_weighted_logistic_regression(X_train, y_train, steps=3500, lr=0.04)

    _, val_probs = predict_logistic(X_val, model, threshold=0.5)
    frame_threshold, val_metrics = choose_threshold(y_val, val_probs, min_recall=0.90)
    threshold, val_alert_metrics = choose_alert_threshold(validation_data, model, min_event_recall=0.90)

    print(f"\nSelected threshold: {threshold:.3f}")
    print(
        "Validation target: event_recall>=0.90, then minimize false alerts "
        f"(event_recall={val_alert_metrics['event_recall']:.3f}, "
        f"alert_precision={val_alert_metrics['alert_precision']:.3f})"
    )
    print(
        f"Frame-oriented threshold would be {frame_threshold:.3f} "
        f"(frame recall={val_metrics['recall']:.3f}, false_alarm={val_metrics['false_alarm']:.3f})"
    )

    y_val_pred = (val_probs >= threshold).astype(np.int64)
    evaluate_predictions(y_val, y_val_pred, "virtual detector validation")
    evaluate_alerts(validation_data, model, threshold, "virtual detector validation")

    y_test_pred, test_probs = predict_logistic(X_test, model, threshold=threshold)
    evaluate_predictions(y_test, y_test_pred, "virtual detector unseen-facility test")
    evaluate_alerts(test_data, model, threshold, "virtual detector unseen-facility test")

    beta, mu, sigma = model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        beta=beta,
        mu=mu,
        sigma=sigma,
        threshold=np.array([threshold], dtype=np.float64),
        window_size=np.array([5], dtype=np.int64),
        feature_count=np.array([X_train.shape[1]], dtype=np.int64),
        use_client_baseline=np.array([int(USE_CLIENT_BASELINE)], dtype=np.int64),
    )
    print(f"\nSaved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
