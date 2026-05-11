import numpy as np
from heterosense import ClientFactory, ConfigurationManager as CM
from heterosense import DatasetBuilder

# =========================
# データ生成
# =========================
clients = ClientFactory.make(3, strategy="round_robin")
cfg = CM.from_clients(clients, n_steps=1000, random_seed=42).to_sim_config()
data = DatasetBuilder(cfg).build()

# =========================
# 簡易分類（zのみ）
# =========================
print("\n--- prediction test (z only) ---")

total = 0
correct = 0

for client_id, bundles in data.items():
    for b in bundles:
        if b.lidar is None:
            continue

        z = np.mean(b.lidar[:, 2])

        # 予測
        if z < 0.2:
            pred = "ABNORMAL"
        else:
            pred = "NORMAL"

        # 正解
        true = "ABNORMAL" if b.semantic_state == "ABNORMAL" else "NORMAL"

        if pred == true:
            correct += 1

        total += 1

print("accuracy:", correct / total)


# =========================
# エラー分析
# =========================
print("\n--- error analysis ---")

error_cases = []

for client_id, bundles in data.items():
    for b in bundles:
        if b.lidar is None:
            continue

        z = np.mean(b.lidar[:, 2])

        if z < 0.2:
            pred = "ABNORMAL"
        else:
            pred = "NORMAL"

        true = "ABNORMAL" if b.semantic_state == "ABNORMAL" else "NORMAL"

        if pred != true:
            error_cases.append((b.semantic_state, float(z)))

# 最初の20件だけ表示
for e in error_cases[:20]:
    print("ERROR:", e)

print("total errors:", len(error_cases))


# =========================
# 点の数の確認（新しい特徴）
# =========================
print("\n--- point count check ---")

for client_id, bundles in data.items():
    for b in bundles[:20]:  # 最初だけ
        if b.lidar is None:
            continue

        z_mean = np.mean(b.lidar[:, 2])
        num_points = len(b.lidar)

        print(
            b.semantic_state,
            "z:", round(z_mean, 3),
            "points:", num_points
        )


# =========================
# 改良版分類（z + point + std）
# =========================
print("\n--- improved prediction (z + point + std) ---")

total = 0
correct = 0

for client_id, bundles in data.items():
    for b in bundles:
        if b.lidar is None:
            continue

        z = b.lidar[:, 2]

        z_mean = np.mean(z)
        z_std = np.std(z)
        num_points = len(z)

        # 改良ルール
        if num_points < 50:
            pred = "ABSENT"
        elif z_mean < 0.2 and z_std < 0.05:
            pred = "ABNORMAL"
        else:
            pred = "NORMAL"

        # 正解
        if b.semantic_state == "ABNORMAL":
            true = "ABNORMAL"
        elif b.semantic_state == "ABSENT":
            true = "ABSENT"
        else:
            true = "NORMAL"

        if pred == true:
            correct += 1

        total += 1

print("improved accuracy:", correct / total)
print("\n--- z std check ---")

for client_id, bundles in data.items():
    for b in bundles[:20]:
        if b.lidar is None:
            continue

        z = b.lidar[:, 2]

        z_mean = np.mean(z)
        z_std = np.std(z)

        print(
            b.semantic_state,
            "mean:", round(z_mean, 3),
            "std:", round(z_std, 3)
        )
        print("\n--- abnormal only ---")

for client_id, bundles in data.items():
    for b in bundles:
        if b.lidar is None:
            continue

        if b.semantic_state != "ABNORMAL":
            continue

        z = b.lidar[:, 2]

        z_mean = np.mean(z)
        z_std = np.std(z)

        print(
            "ABNORMAL",
            "mean:", round(z_mean, 3),
            "std:", round(z_std, 3)
        )