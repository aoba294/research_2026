from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.check_data import make_stress_facility_dataset
from experiments.train_cnn_fall_detector import cnn_sequences
from experiments.train_virtual_fall_detector import evaluate_alert_sequences


MODEL_PATH = Path("results/virtual_fall_detector_cnn_5frame_baseline.npz")


def load_cnn_model(path):
    data = np.load(path)
    params = {
        "Wc": data["Wc"],
        "bc": data["bc"],
        "Wd": data["Wd"],
        "bd": data["bd"],
    }
    model = (params, data["mu"], data["sigma"])
    threshold = float(data["threshold"][0])
    use_client_baseline = bool(int(data["use_client_baseline"][0]))
    return model, threshold, use_client_baseline


def main():
    model, threshold, use_client_baseline = load_cnn_model(MODEL_PATH)

    print("Building stress-test dataset...")
    stress_data = make_stress_facility_dataset(n_steps=10000, seed=2026)
    sequences = cnn_sequences(stress_data, model, use_client_baseline)

    print(f"Loaded model: {MODEL_PATH}")
    print(f"threshold: {threshold:.3f}")
    print(f"use_client_baseline: {use_client_baseline}")

    evaluate_alert_sequences(
        sequences,
        threshold,
        "1D-CNN with facility baseline: stress-facility test",
    )

    print("\nThreshold sensitivity:")
    for thr in [0.75, 0.80, 0.85, 0.90, 0.94, threshold]:
        m = evaluate_alert_sequences(
            sequences,
            thr,
            "stress threshold",
            verbose=False,
        )
        print(
            f"threshold={thr:.3f} "
            f"event_recall={m['event_recall']:.3f} "
            f"alert_precision={m['alert_precision']:.3f} "
            f"alerts={m['alerts']} "
            f"false_alerts={m['false_alerts']}"
        )


if __name__ == "__main__":
    main()
