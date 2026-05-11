import numpy as np
from heterosense import ClientFactory, ConfigurationManager as CM
from heterosense import DatasetBuilder

# データ生成
clients = ClientFactory.make(3, strategy="round_robin")
cfg = CM.from_clients(clients, n_steps=1000, random_seed=42).to_sim_config()
data = DatasetBuilder(cfg).build()

# 確認
for client_id, bundles in data.items():
    print(f"\nclient {client_id}, steps={len(bundles)}")

    for b in bundles[:3]:
        print("state:", b.semantic_state)

        if b.lidar is not None:
            print("lidar shape:", b.lidar.shape)

# 🔥 ここからSTEP4（同じ階層に書く）
print("\n--- z mean check ---")

for client_id, bundles in data.items():
    for b in bundles:
        if b.lidar is None:
            continue

        z = b.lidar[:, 2]
        print(b.semantic_state, np.mean(z))
        print("\n--- prediction test ---")

total = 0
correct = 0

for client_id, bundles in data.items():
    for b in bundles:
        if b.lidar is None:
            continue

        # 特徴量
        z = np.mean(b.lidar[:, 2])

        # 予測（ルール）
        if z < 0.2:
            pred = "ABNORMAL"
        else:
            pred = "NORMAL"

        # 正解（ラベル）
        if b.semantic_state == "ABNORMAL":
            true = "ABNORMAL"
        else:
            true = "NORMAL"

        # 比較
        if pred == true:
            correct += 1

        total += 1

print("accuracy:", correct / total)