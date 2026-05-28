from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.check_data import (
    compute_client_baselines,
    evaluate_predictions,
    extract_window_features,
    make_dataset,
    make_unseen_facility_dataset,
)
from experiments.train_virtual_fall_detector import (
    alert_indices_from_probs,
    evaluate_alert_sequences,
    fall_episodes,
    metrics,
    window_features_for_bundles,
)


OUT_BASELINE = Path("results/virtual_fall_detector_nn_5frame_baseline.npz")
OUT_NO_BASELINE = Path("results/virtual_fall_detector_nn_5frame_no_baseline.npz")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def relu(x):
    return np.maximum(0.0, x)


def make_mlp(input_dim, hidden_dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1)),
        "b2": np.zeros(1),
    }


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    h1 = relu(z1)
    logits = (h1 @ params["W2"] + params["b2"]).ravel()
    probs = sigmoid(logits)
    return z1, h1, probs


def fit_weighted_mlp(X_train, y_train, hidden_dim=32, epochs=1400, lr=0.01, seed=0):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X_train - mu) / sigma
    y = y_train.astype(np.float64)

    params = make_mlp(X.shape[1], hidden_dim=hidden_dim, seed=seed)
    opt = {name: np.zeros_like(value) for name, value in params.items()}

    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = {name: np.zeros_like(value) for name, value in params.items()}
    v = {name: np.zeros_like(value) for name, value in params.items()}

    for epoch in range(1, epochs + 1):
        z1, h1, probs = forward(X, params)
        dlogits = ((probs - y) * weights / len(y))[:, None]

        grads = {
            "W2": h1.T @ dlogits,
            "b2": np.sum(dlogits, axis=0),
        }
        dh1 = dlogits @ params["W2"].T
        dz1 = dh1 * (z1 > 0)
        grads["W1"] = X.T @ dz1
        grads["b1"] = np.sum(dz1, axis=0)

        for name in params:
            m[name] = beta1 * m[name] + (1 - beta1) * grads[name]
            v[name] = beta2 * v[name] + (1 - beta2) * (grads[name] ** 2)
            m_hat = m[name] / (1 - beta1 ** epoch)
            v_hat = v[name] / (1 - beta2 ** epoch)
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return params, mu, sigma


def predict_mlp(X, model, threshold=0.5):
    params, mu, sigma = model
    Xs = (X - mu) / sigma
    _, _, probs = forward(Xs, params)
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


def nn_alert_sequences(data, model, use_client_baseline):
    sequences = []
    baselines = compute_client_baselines(data) if use_client_baseline else {}
    for client_id, bundles in data.items():
        X, indices, labels = window_features_for_bundles(
            bundles,
            baseline=baselines.get(client_id),
        )
        if len(X) == 0:
            continue
        _, probs = predict_mlp(X, model, threshold=0.5)
        sequences.append((indices, labels, probs))
    return sequences


def choose_alert_threshold(data, model, use_client_baseline, min_event_recall=0.90):
    sequences = nn_alert_sequences(data, model, use_client_baseline)
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
        W1=params["W1"],
        b1=params["b1"],
        W2=params["W2"],
        b2=params["b2"],
        mu=mu,
        sigma=sigma,
        threshold=np.array([threshold], dtype=np.float64),
        window_size=np.array([5], dtype=np.int64),
        use_client_baseline=np.array([int(use_client_baseline)], dtype=np.int64),
    )


def run_experiment(train_data, validation_data, test_data, use_client_baseline, out_path):
    label = "NN with facility baseline" if use_client_baseline else "NN without facility baseline"
    print(f"\n=== {label} ===")

    X_train, y_train = extract_window_features(
        train_data,
        use_client_baseline=use_client_baseline,
    )
    X_val, y_val = extract_window_features(
        validation_data,
        use_client_baseline=use_client_baseline,
    )
    X_test, y_test = extract_window_features(
        test_data,
        use_client_baseline=use_client_baseline,
    )

    print(f"train samples={len(y_train)} falls={int(y_train.sum())} features={X_train.shape[1]}")
    print(f"validation samples={len(y_val)} falls={int(y_val.sum())}")
    print(f"test samples={len(y_test)} falls={int(y_test.sum())}")

    model = fit_weighted_mlp(X_train, y_train, hidden_dim=32, epochs=1400, lr=0.01, seed=7)

    _, val_probs = predict_mlp(X_val, model, threshold=0.5)
    frame_threshold, frame_metrics = choose_frame_threshold(y_val, val_probs, min_recall=0.90)
    threshold, alert_metrics = choose_alert_threshold(
        validation_data,
        model,
        use_client_baseline=use_client_baseline,
        min_event_recall=0.90,
    )

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
        nn_alert_sequences(validation_data, model, use_client_baseline),
        threshold,
        f"{label}: validation",
    )

    test_pred, _ = predict_mlp(X_test, model, threshold=threshold)
    evaluate_predictions(y_test, test_pred, f"{label}: unseen-facility test frame-level")
    test_alerts = evaluate_alert_sequences(
        nn_alert_sequences(test_data, model, use_client_baseline),
        threshold,
        f"{label}: unseen-facility test",
    )

    save_model(out_path, model, threshold, use_client_baseline)
    print(f"saved model: {out_path}")
    return test_alerts


def main():
    print("Building datasets for NN comparison...")
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

    print("\n=== NN comparison summary: unseen-facility event-level ===")
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
