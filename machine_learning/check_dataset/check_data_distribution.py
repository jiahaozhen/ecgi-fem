import os
import h5py
import numpy as np
from collections import Counter
from collections import defaultdict

if __name__ == "__main__":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/v_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/v_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/v_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/v_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/v_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/v_dataset/",
    ]

    # case_name -> list of y arrays
    case_y = defaultdict(list)

    # ---------- 收集 ----------
    for d in data_dir:
        assert os.path.isdir(d), f"{d} not found"

        # 提取 case 名称（normal_male / normal_male2）
        case_name = d.split(os.sep)[-4]

        for f in os.listdir(d):
            if f.endswith(".h5"):
                with h5py.File(os.path.join(d, f), "r") as data:
                    case_y[case_name].append(data["y"][:])  # (N, L)

    # ---------- 统计 ----------
    print("=" * 90)
    print("Per-case (subject-level) multi-label statistics")
    print("=" * 90)

    for case, y_list in case_y.items():
        y = np.concatenate(y_list, axis=0)  # (N_total, L)

        N, L = y.shape
        label_counts = y.sum(axis=0)
        positive_samples = np.sum(y.sum(axis=1) > 0)
        avg_labels = y.sum() / N

        print(f"\n🧑 Case: {case}")
        print(f"   total samples: {N}")
        print(f"   positive samples: {positive_samples}")
        print(f"   avg labels per sample: {avg_labels:.3f}")

        # 单标签统计
        for i, cnt in enumerate(label_counts):
            print(f"   Label {i:02d}: {int(cnt)} samples")

        # ---------- 标签数量分布 ----------
        label_nums = y.sum(axis=1)  # 每个样本有多少个正标签

        num_no_label = np.sum(label_nums == 0)
        num_single_label = np.sum(label_nums == 1)
        num_multi_label = np.sum(label_nums >= 2)

        print("\n📊 Label cardinality distribution")
        print(f"   无标签样本 (0 label): {num_no_label}")
        print(f"   单标签样本 (1 label): {num_single_label}")
        print(f"   多标签样本 (>=2 labels): {num_multi_label}")

        print(f"   占比:")
        print(f"     0 label : {num_no_label / N:.2%}")
        print(f"     1 label : {num_single_label / N:.2%}")
        print(f"     ≥2 label: {num_multi_label / N:.2%}")

        # # # ========= 新增：统计 label 组合 =========
        # combo_counter = Counter()
        # for row in y:
        #     combo = "".join(map(str, row.astype(int)))
        #     combo_counter[combo] += 1

        # print(f"\n🧑 Case: {case}")
        # print(f"   不同 label 组合数: {len(combo_counter)}")

        # for combo, cnt in combo_counter.most_common():
        #     print(f"{combo} -> {cnt}")
