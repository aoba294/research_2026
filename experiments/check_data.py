import numpy as np
import matplotlib.pyplot as plt
from heterosense import ClientFactory, ConfigurationManager as CM
from heterosense import DatasetBuilder


def make_dataset(n_steps=1000, seed=42, abnormal_rate=None):
    clients = ClientFactory.make(3, strategy="round_robin")
    if abnormal_rate is not None:
        for client in clients:
            client["abnormal_rate"] = abnormal_rate
    cfg = CM.from_clients(clients, n_steps=n_steps, random_seed=seed).to_sim_config()
    return DatasetBuilder(cfg).build()


def make_unseen_facility_dataset(n_steps=10000, seed=42, abnormal_rate=None):
    clients = [
        {
            "client_id": "unseen_small_room",
            "channel_availability": ["lidar", "bed"],
            "room_width": 3.5,
            "room_height": 4.0,
            "sensor_noise_level": 2.0,
            "abnormal_rate": 0.006,
            "bed_position": [1.4, 2.6],
            "bed_radius": 0.7,
            "lidar_occlusion": 0.25,
        },
        {
            "client_id": "unseen_occluded_lidar",
            "channel_availability": ["lidar"],
            "room_width": 7.0,
            "room_height": 5.5,
            "sensor_noise_level": 4.0,
            "abnormal_rate": 0.004,
            "lidar_occlusion": 0.45,
        },
        {
            "client_id": "unseen_quiet_bed",
            "channel_availability": ["lidar", "bed"],
            "room_width": 5.5,
            "room_height": 6.0,
            "sensor_noise_level": 1.6,
            "abnormal_rate": 0.003,
            "bed_position": [4.0, 1.6],
            "bed_radius": 0.9,
            "lidar_occlusion": 0.15,
        },
    ]

    if abnormal_rate is not None:
        for client in clients:
            client["abnormal_rate"] = abnormal_rate

    cfg = CM.from_clients(clients, n_steps=n_steps, random_seed=seed).to_sim_config()
    return DatasetBuilder(cfg).build()


def make_stress_facility_dataset(n_steps=10000, seed=42, abnormal_rate=None):
    clients = [
        {
            "client_id": "stress_cluttered_room",
            "channel_availability": ["lidar", "bed"],
            "room_width": 3.2,
            "room_height": 6.8,
            "sensor_noise_level": 4.5,
            "abnormal_rate": 0.005,
            "bed_position": [0.8, 5.8],
            "bed_radius": 0.65,
            "lidar_occlusion": 0.55,
        },
        {
            "client_id": "stress_lidar_only_far",
            "channel_availability": ["lidar"],
            "room_width": 8.0,
            "room_height": 7.0,
            "sensor_noise_level": 5.5,
            "abnormal_rate": 0.004,
            "lidar_occlusion": 0.65,
        },
        {
            "client_id": "stress_bed_shifted",
            "channel_availability": ["lidar", "bed"],
            "room_width": 6.5,
            "room_height": 3.8,
            "sensor_noise_level": 3.8,
            "abnormal_rate": 0.006,
            "bed_position": [5.8, 0.7],
            "bed_radius": 1.0,
            "lidar_occlusion": 0.40,
        },
    ]

    if abnormal_rate is not None:
        for client in clients:
            client["abnormal_rate"] = abnormal_rate

    cfg = CM.from_clients(clients, n_steps=n_steps, random_seed=seed).to_sim_config()
    return DatasetBuilder(cfg).build()


def true_binary_label(bundle):
    return "ABNORMAL" if bundle.semantic_state == "ABNORMAL" else "NORMAL"


def predict_z_only(bundle):
    z_mean = float(np.mean(bundle.lidar[:, 2]))
    return "ABNORMAL" if z_mean < 0.2 else "NORMAL"


def predict_z_point_std(bundle):
    z = bundle.lidar[:, 2]
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))
    num_points = len(z)

    if num_points < 50:
        return "ABSENT"
    if z_mean < 0.2 and z_std < 0.05:
        return "ABNORMAL"
    return "NORMAL"


def evaluate_binary(data, predictor, title):
    tp = tn = fp = fn = 0

    for bundles in data.values():
        for b in bundles:
            if b.lidar is None:
                continue

            pred = predictor(b)
            if pred == "ABSENT":
                pred = "NORMAL"
            true = true_binary_label(b)

            if true == "ABNORMAL" and pred == "ABNORMAL":
                tp += 1
            elif true == "NORMAL" and pred == "NORMAL":
                tn += 1
            elif true == "NORMAL" and pred == "ABNORMAL":
                fp += 1
            elif true == "ABNORMAL" and pred == "NORMAL":
                fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    fall_recall = tp / (tp + fn) if (tp + fn) else 0.0
    fall_precision = tp / (tp + fp) if (tp + fp) else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n--- {title} ---")
    print(f"accuracy:             {accuracy:.3f}")
    print(f"fall recall:          {fall_recall:.3f}  (detected falls / true falls)")
    print(f"fall precision:       {fall_precision:.3f}  (true falls / predicted falls)")
    print(f"false alarm rate:     {false_alarm_rate:.3f}  (normal frames predicted as falls)")
    print("confusion matrix:")
    print("                 pred NORMAL   pred ABNORMAL")
    print(f"true NORMAL      {tn:11d}   {fp:13d}")
    print(f"true ABNORMAL    {fn:11d}   {tp:13d}")


def evaluate_predictions(y_true, y_pred, title):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    fall_recall = tp / (tp + fn) if (tp + fn) else 0.0
    fall_precision = tp / (tp + fp) if (tp + fp) else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n--- {title} ---")
    print(f"accuracy:             {accuracy:.3f}")
    print(f"fall recall:          {fall_recall:.3f}  (detected falls / true falls)")
    print(f"fall precision:       {fall_precision:.3f}  (true falls / predicted falls)")
    print(f"false alarm rate:     {false_alarm_rate:.3f}  (normal frames predicted as falls)")
    print("confusion matrix:")
    print("                 pred NORMAL   pred ABNORMAL")
    print(f"true NORMAL      {tn:11d}   {fp:13d}")
    print(f"true ABNORMAL    {fn:11d}   {tp:13d}")


def extract_frame_features(data):
    x_rows = []
    y_rows = []

    for bundles in data.values():
        for b in bundles:
            if b.lidar is None:
                continue

            pts = b.lidar
            z = pts[:, 2]
            x_rows.append([
                float(np.mean(z)),
                float(np.std(z)),
                float(np.min(z)),
                float(np.max(z)),
                float(np.mean(z < 0.2)),
                float(np.mean(z < 0.4)),
                float(len(z)),
                float(np.std(pts[:, 0])),
                float(np.std(pts[:, 1])),
                float(np.linalg.norm([np.std(pts[:, 0]), np.std(pts[:, 1])])),
            ])
            y_rows.append(1 if b.semantic_state == "ABNORMAL" else 0)

    return np.array(x_rows, dtype=np.float64), np.array(y_rows, dtype=np.int64)


def _lidar_frame_features(bundle):
    if bundle.lidar is None:
        return None

    pts = bundle.lidar
    z = pts[:, 2]
    return np.array([
        float(np.mean(z)),
        float(np.std(z)),
        float(np.min(z)),
        float(np.max(z)),
        float(np.mean(z < 0.2)),
        float(np.mean(z < 0.4)),
        float(len(z)),
        float(np.std(pts[:, 0])),
        float(np.std(pts[:, 1])),
        float(np.linalg.norm([np.std(pts[:, 0]), np.std(pts[:, 1])])),
    ], dtype=np.float64)


def compute_client_baselines(data):
    baselines = {}
    for client_id, bundles in data.items():
        rows = []
        for b in bundles:
            if b.lidar is None or b.semantic_state == "ABNORMAL":
                continue
            features = _lidar_frame_features(b)
            pressure = float(b.pressure.sum()) if b.pressure is not None else 0.0
            if features is not None:
                rows.append(np.r_[features, pressure])
        if rows:
            baselines[client_id] = np.mean(np.vstack(rows), axis=0)
    return baselines


def extract_window_features(data, window_size=5, use_client_baseline=False):
    x_rows = []
    y_rows = []
    center = window_size // 2
    baselines = compute_client_baselines(data) if use_client_baseline else {}

    for client_id, bundles in data.items():
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
            if use_client_baseline and client_id in baselines:
                feature_parts.append(center_with_pressure - baselines[client_id])
            window_features = np.concatenate(feature_parts)
            x_rows.append(window_features)
            y_rows.append(1 if window[center].semantic_state == "ABNORMAL" else 0)

    return np.array(x_rows, dtype=np.float64), np.array(y_rows, dtype=np.int64)


def train_test_split_stratified(X, y, test_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for label in [0, 1]:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_weighted_logistic_regression(X_train, y_train, steps=2500, lr=0.05):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X_train - mu) / sigma
    Xb = np.c_[np.ones(len(Xs)), Xs]

    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    weights = np.where(y_train == 1, len(y_train) / (2 * pos), len(y_train) / (2 * neg))

    beta = np.zeros(Xb.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = np.clip(Xb @ beta, -40, 40)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = Xb.T @ ((probs - y_train) * weights) / len(y_train)
        beta -= lr * grad

    return beta, mu, sigma


def predict_logistic(X, model, threshold=0.5):
    beta, mu, sigma = model
    Xs = (X - mu) / sigma
    Xb = np.c_[np.ones(len(Xs)), Xs]
    probs = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -40, 40)))
    return (probs >= threshold).astype(np.int64), probs


def evaluate_simple_ml(train_data, test_data, title, feature_extractor=extract_frame_features):
    X_train, y_train = feature_extractor(train_data)
    X_test, y_test = feature_extractor(test_data)
    model = fit_weighted_logistic_regression(X_train, y_train)
    y_pred, probs = predict_logistic(X_test, model, threshold=0.5)

    print(f"\n--- dataset for {title} ---")
    print(f"train samples: {len(y_train)}  falls: {int(y_train.sum())}")
    print(f"test samples:  {len(y_test)}  falls: {int(y_test.sum())}")
    evaluate_predictions(y_test, y_pred, f"{title}: threshold 0.5")

    y_pred_sensitive, _ = predict_logistic(X_test, model, threshold=0.3)
    evaluate_predictions(y_test, y_pred_sensitive, f"{title}: threshold 0.3")


def print_feature_summary(data):
    rows = []
    for bundles in data.values():
        for b in bundles:
            if b.lidar is None:
                continue
            z = b.lidar[:, 2]
            rows.append((
                b.semantic_state,
                float(np.mean(z)),
                float(np.std(z)),
                len(z),
            ))

    print("\n--- z feature summary ---")
    for state in sorted(set(row[0] for row in rows)):
        vals = np.array([(m, s, n) for st, m, s, n in rows if st == state])
        print(
            f"{state:10s}",
            f"n={len(vals):4d}",
            f"z_mean={vals[:, 0].mean():.3f}",
            f"z_std={vals[:, 1].mean():.3f}",
            f"z_std_range=({vals[:, 1].min():.3f}, {vals[:, 1].max():.3f})",
            f"points={vals[:, 2].mean():.1f}",
        )


def plot_z_features(data):
    points_by_state = {}

    for bundles in data.values():
        for b in bundles:
            if b.lidar is None:
                continue

            z = b.lidar[:, 2]
            z_mean = np.mean(z)
            z_std = np.std(z)
            points_by_state.setdefault(b.semantic_state, []).append((z_mean, z_std))

    plt.figure(figsize=(8, 6))

    for semantic_state, points in points_by_state.items():
        points = np.array(points)
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            alpha=0.7,
            label=semantic_state,
        )

    plt.axvline(x=0.2, color="red", linestyle="--", linewidth=1.5, label="z_mean = 0.2")
    plt.axhline(y=0.05, color="blue", linestyle="--", linewidth=1.5, label="z_std = 0.05")

    plt.xlabel("z_mean")
    plt.ylabel("z_std")
    plt.title("LiDAR z_mean and z_std by semantic_state")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    eval_data = make_dataset(n_steps=10000, seed=42)
    unseen_eval_data = make_unseen_facility_dataset(n_steps=10000, seed=42)
    train_realistic_data = make_dataset(n_steps=10000, seed=123)
    train_fall_enriched_data = make_dataset(n_steps=10000, seed=123, abnormal_rate=0.08)

    print("\n=== realistic evaluation data ===")
    evaluate_binary(eval_data, predict_z_only, "prediction test: z only")
    evaluate_binary(eval_data, predict_z_point_std, "prediction test: z + point + std")

    evaluate_simple_ml(
        train_realistic_data,
        eval_data,
        "simple ML trained on realistic data",
    )
    evaluate_simple_ml(
        train_fall_enriched_data,
        eval_data,
        "frame ML trained on fall-enriched data",
        feature_extractor=extract_frame_features,
    )
    evaluate_simple_ml(
        train_fall_enriched_data,
        eval_data,
        "5-frame ML trained on fall-enriched data",
        feature_extractor=extract_window_features,
    )

    print("\n=== unseen facility evaluation data ===")
    evaluate_binary(unseen_eval_data, predict_z_only, "prediction test: z only")
    evaluate_binary(unseen_eval_data, predict_z_point_std, "prediction test: z + point + std")
    evaluate_simple_ml(
        train_fall_enriched_data,
        unseen_eval_data,
        "frame ML trained on fall-enriched data, tested on unseen facilities",
        feature_extractor=extract_frame_features,
    )
    evaluate_simple_ml(
        train_fall_enriched_data,
        unseen_eval_data,
        "5-frame ML trained on fall-enriched data, tested on unseen facilities",
        feature_extractor=extract_window_features,
    )

    print_feature_summary(eval_data)
    plot_z_features(eval_data)


if __name__ == "__main__":
    main()
