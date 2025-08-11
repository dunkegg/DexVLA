
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def compute_turning_degree(x, z):
    dx = np.diff(x)
    dz = np.diff(z)
    directions = np.arctan2(dz, dx)
    angle_changes = np.diff(directions)
    angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
    total_angle_change = np.sum(np.abs(angle_changes))
    average_angle_change = np.mean(np.abs(angle_changes)) if len(angle_changes) > 0 else 0
    return total_angle_change, average_angle_change

def plot_all_rel_paths(h5_path, output_png):
    with h5py.File(h5_path, "r") as f:
        if "follow_paths" not in f:
            print(f"❌ {h5_path} 中未找到 'follow_paths'")
            return

        all_keys = sorted(f["follow_paths"].keys())
        num = len(all_keys)

        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap("tab20", num)

        for i, key in enumerate(all_keys):
            path = f["follow_paths"][key]
            if "rel_path" not in path:
                continue

            rel_path = path["rel_path"][:]
            if rel_path.shape[1] < 3:
                print(f"⚠️ 跳过 {key}，rel_path 列数不足")
                continue

            x = rel_path[:, 0]
            z = rel_path[:, 2]

            total_angle, avg_angle = compute_turning_degree(x, z)

            # ==== 计算首尾连线方向角 ====
            vec_x = x[-1] - x[0]
            vec_z = z[-1] - z[0]
            angle_to_z = np.arctan2(vec_x, vec_z)  # z轴是(0,1)，所以是 arctan2(x, z)
            angle_to_z_deg = np.degrees(angle_to_z)

            label = f"{key} | ∑θ={total_angle:.2f}, θ̄={avg_angle:.3f}"

            plt.plot(x, z, label=label, color=cmap(i % 20))

            # 标注夹角（在终点附近）
            plt.text(x[-1], z[-1], f"{angle_to_z_deg:.1f}°", fontsize=7,
                     color=cmap(i % 20), ha='center', va='bottom')

            print(f"[{Path(h5_path).stem}::{key}] 总转角: {total_angle:.2f} rad, 平均: {avg_angle:.4f} rad, 与z轴夹角: {angle_to_z_deg:.1f}°")

        plt.title(f"Trajectories in {Path(h5_path).name}")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend(fontsize=7, loc='best', ncol=2)
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=200)
        plt.close()
        print(f"✅ 保存至: {output_png}\n")


def batch_process_hdf5_folder(folder_path, output_dir):
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    h5_files = sorted(folder_path.glob("*.hdf5"))
    if not h5_files:
        print("⚠️ 没有找到 HDF5 文件在目录：", folder_path)
        return

    for h5_path in h5_files:
        filename = h5_path.stem + ".png"
        output_path = output_dir / filename
        plot_all_rel_paths(h5_path, output_path)

# 示例调用
if __name__ == "__main__":
    input_dir = "data/proc_data/single_follow"     # ← 包含 .hdf5 文件的目录
    output_dir = "plots"             # ← 图像输出保存目录
    batch_process_hdf5_folder(input_dir, output_dir)
