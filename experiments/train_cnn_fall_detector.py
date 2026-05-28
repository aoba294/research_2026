from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.check_data import (
    compute_client_baselines,
    evaluate_predictions,
    make_dataset,
    make_unseen_facility_dataset,
    _lidar_frame_features,
)
from experiments.train_virtual_fall_detector import (
    evaluate_alert_sequences,
    metrics,
)


OUT_BASELINE = Path("results/virtual_fall_detector_cnn_5frame_baseline.npz")
OUT_NO_BASELINE = Path("results/virtual_fall_detector_cnn_5frame_no_baseline.npz")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def relu(x):
    return np.maximum(0.0, x)


def frame_features_with_pressure(bundle):
    f = _lidar_frame_features(bundle)
    if f is None:
        return None
    pressure = float(bundle.pressure.sum()) if bundle.pressure is not None else 0.0
    return np.r_[f, pressure]


def extract_cnn_windows(data, window_size=5, use_client_baseline=False):
    X_rows = []
    y_rows = []
    center = window_size // 2
    baselines = compute_client_baselines(data) if use_client_baseline else {}

    for client_id, bundles in data.items():
        baseline = baselines.get(client_id)
        for start in range(0, len(bundles) - window_size + 1):
            window = bundles[start:start + window_size]
            per_frame = [frame_features_with_pressure(b) for b in window]
            if any(f is None for f in per_frame):
                continue
            F = np.vstack(per_frame)
            if use_client_baseline and baseline is not None:
                F = np.concatenate([F, F - baseline], axis=1)
            X_rows.append(F)
            y_rows.append(1 if window[center].semantic_state == "ABNORMAL" else 0)

    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.int64)


def cnn_sequences(data, model, use_client_baseline, window_size=5):
    sequences = []
    center = window_size // 2
    baselines = compute_client_baselines(data) if use_client_baseline else {}

    for client_id, bundles in data.items():
        X_rows = []
        indices = []
        labels = []
        baseline = baselines.get(client_id)

        for start in range(0, len(bundles) - window_size + 1):
            window = bundles[start:start + window_size]
            per_frame = [frame_features_with_pressure(b) for b in window]
            if any(f is None for f in per_frame):
                continue
            F = np.vstack(per_frame)
            if use_client_baseline and baseline is not None:
                F = np.concatenate([F, F - baseline], axis=1)
            X_rows.append(F)
            indices.append(start + center)
            labels.append(1 if window[center].semantic_state == "ABNORMAL" else 0)

        if X_rows:
            _, probs = predict_cnn(np.array(X_rows, dtype=np.float64), model)
            sequences.append((
                np.array(indices, dtype=np.int64),
                np.array(labels, dtype=np.int64),
                probs,
            ))
    return sequences


def make_cnn(input_dim, filters=24, kernel=3, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "Wc": rng.normal(0.0, np.sqrt(2.0 / (kernel * input_dim)), size=(kernel, input_dim, filters)),
        "bc": np.zeros(filters),
        "Wd": rng.normal(0.0, np.sqrt(2.0 / filters), size=(filters, 1)),
        "bd": np.zeros(1),
    }


def sliding_windows_time(X, kernel):
    # X: (N, T, D) -> (N, T-kernel+1, kernel, D)
    N, T, D = X.shape
    return np.stack([X[:, i:i + kernel, :] for i in range(T - kernel + 1)], axis=1)


def forward_cnn(X, params):
    kernel = params["Wc"].shape[0]
    Xw = sliding_windows_time(X, kernel)
    conv = np.einsum("nlkd,kdf->nlf", Xw, params["Wc"]) + params["bc"]
    act = relu(conv)
    max_idx = np.argmax(act, axis=1)
    pooled = np.max(act, axis=1)
    logits = (pooled @ params["Wd"] + params["bd"]).ravel()
    probs = sigmoid(logits)
    cache = (Xw, conv, act, max_idx, pooled, probs)
    return probs, cache


def fit_weighted_cnn(X_train, y_train, filters=16, epochs=300, lr=0.012, seed=0):
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sigma = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X_train - mu) / sigma
    y = y_train.astype(np.float64)

    params = make_cnn(X.shape[-1], filters=filters, kernel=3, seed=seed)
    m = {name: np.zeros_like(value) for name, value in params.items()}
    v = {name: np.zeros_like(value) for name, value in params.items()}

    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, epochs + 1):
        probs, cache = forward_cnn(X, params)
        Xw, conv, act, max_idx, pooled, _ = cache

        dlogits = ((probs - y) * weights / len(y))[:, None]
        grads = {
            "Wd": pooled.T @ dlogits,
            "bd": np.sum(dlogits, axis=0),
        }

        dpooled = dlogits @ params["Wd"].T
        dact = np.zeros_like(act)
        for n in range(len(X)):
            dact[n, max_idx[n], np.arange(act.shape[2])] = dpooled[n]
        dconv = dact * (conv > 0)

        grads["Wc"] = np.einsum("nlkd,nlf->kdf", Xw, dconv)
        grads["bc"] = np.sum(dconv, axis=(0, 1))

        for name in params:
            m[name] = beta1 * m[name] + (1 - beta1) * grads[name]
            v[name] = beta2 * v[name] + (1 - beta2) * (grads[name] ** 2)
            m_hat = m[name] / (1 - beta1 ** epoch)
            v_hat = v[name] / (1 - beta2 ** epoch)
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return params, mu, sigma


def predict_cnn(X, model, threshold=0.5):
    params, mu, sigma = model
    Xs = (X - mu) / sigma
    probs, _ = forward_cnn(Xs, params)
    return (probs >= threshold).astype(np.int64), probs


def choose_frame_threshold(y_true, probs, min_recall=0.90):
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (probs >= threshold).astype(np.int64)
        m = metrics(y_true, pred)
        if m["recall"] < min_recall:
            continue
        candidate = (m["false_alarm"], -m["precision"], -m["accuracy"], threshold, m)
        if best is None or candidate < best:
            best = candidate
    if best is not None:
        return float(best[3]), best[4]
    return 0.5, metrics(y_true, (probs >= 0.5).astype(np.int64))


def choose_alert_threshold(data, model, use_client_baseline, min_event_recall=0.90):
    sequences = cnn_sequences(data, model, use_client_baseline)
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        m = evaluate_alert_sequences(sequences, threshold, "threshold search", verbose=False)
        if m["event_recall"] < min_event_recall:
            continue
        candidate = (m["false_alerts"], -m["alert_precision"], -m["event_recall"], threshold, m)
        if best is None or candidate < best:
            best = candidate
    if best is not None:
        return float(best[3]), best[4]
    return 0.5, evaluate_alert_sequences(sequences, 0.5, "threshold search", verbose=False)


def save_model(path, model, threshold, use_client_baseline):
    params, mu, sigma = model
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        Wc=params["Wc"],
        bc=params["bc"],
        Wd=params["Wd"],
        bd=params["bd"],
        mu=mu,
        sigma=sigma,
        threshold=np.array([threshold], dtype=np.float64),
        window_size=np.array([5], dtype=np.int64),
        use_client_baseline=np.array([int(use_client_baseline)], dtype=np.int64),
    )


def run_experiment(train_data, validation_data, test_data, use_client_baseline, out_path):
    label = "1D-CNN with facility baseline" if use_client_baseline else "1D-CNN without facility baseline"
    print(f"\n=== {label} ===")

    X_train, y_train = extract_cnn_windows(train_data, use_client_baseline=use_client_baseline)
    X_val, y_val = extract_cnn_windows(validation_data, use_client_baseline=use_client_baseline)
    X_test, y_test = extract_cnn_windows(test_data, use_client_baseline=use_client_baseline)

    print(f"train samples={len(y_train)} falls={int(y_train.sum())} shape={X_train.shape}")
    print(f"validation samples={len(y_val)} falls={int(y_val.sum())}")
    print(f"test samples={len(y_test)} falls={int(y_test.sum())}")

    model = fit_weighted_cnn(X_train, y_train, filters=16, epochs=300, lr=0.012, seed=11)

    _, val_probs = predict_cnn(X_val, model)
    frame_threshold, frame_metrics = choose_frame_threshold(y_val, val_probs, min_recall=0.90)
    threshold, alert_metrics = choose_alert_threshold(validation_data, model, use_client_baseline)

    print(f"selected alert threshold={threshold:.3f}")
    print(
        f"validation event_recall={alert_metrics['event_recall']:.3f}, "
        f"alert_precision={alert_metrics['alert_precision']:.3f}"
    )
    print(
        f"frame-oriented threshold={frame_threshold:.3f}, "
        f"frame recall={frame_metrics['recall']:.3f}, false_alarm={frame_metrics['false_alarm']:.3f}"
    )

    val_pred = (val_probs >= threshold).astype(np.int64)
    evaluate_predictions(y_val, val_pred, f"{label}: validation frame-level")
    evaluate_alert_sequences(
        cnn_sequences(validation_data, model, use_client_baseline),
        threshold,
        f"{label}: validation",
    )

    test_pred, _ = predict_cnn(X_test, model, threshold=threshold)
    evaluate_predictions(y_test, test_pred, f"{label}: unseen-facility test frame-level")
    test_alerts = evaluate_alert_sequences(
        cnn_sequences(test_data, model, use_client_baseline),
        threshold,
        f"{label}: unseen-facility test",
    )

    save_model(out_path, model, threshold, use_client_baseline)
    print(f"saved model: {out_path}")
    return test_alerts


def main():
    print("Building datasets for 1D-CNN comparison...")
    train_data = make_dataset(n_steps=12000, seed=123, abnormal_rate=0.08)
    validation_data = make_unseen_facility_dataset(n_steps=8000, seed=7)
    test_data = make_unseen_facility_dataset(n_steps=10000, seed=42)

    no_base = run_experiment(
        train_data,
        validation_data,
        test_data,
        use_client_baseline=False,
        out_path=OUT_NO_BASELINE,
    )
    with_base = run_experiment(
        train_data,
        validation_data,
        test_data,
        use_client_baseline=True,
        out_path=OUT_BASELINE,
    )

    print("\n=== 1D-CNN comparison summary: unseen-facility event-level ===")
    print(
        "without baseline: "
        f"event_recall={no_base['event_recall']:.3f}, "
        f"alert_precision={no_base['alert_precision']:.3f}, "
        f"false_alerts={no_base['false_alerts']}"
    )
    print(
        "with baseline:    "
        f"event_recall={with_base['event_recall']:.3f}, "
        f"alert_precision={with_base['alert_precision']:.3f}, "
        f"false_alerts={with_base['false_alerts']}"
    )


if __name__ == "__main__":
    main()
