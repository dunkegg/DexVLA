import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

# ==== 设置阈值 ====
THRESH_TOTAL_ANGLE = 1.49
THRESH_AVG_ANGLE = 0.2

def compute_turning_degree(x, z):
    dx = np.diff(x)
    dz = np.diff(z)
    directions = np.arctan2(dz, dx)
    angle_changes = np.diff(directions)
    angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
    total_angle_change = np.sum(np.abs(angle_changes))
    avg_angle_change = np.mean(np.abs(angle_changes)) if len(angle_changes) > 0 else 0
    return total_angle_change, avg_angle_change

def plot_single_trajectory(x, z, file_stem, traj_id, save_dir, total_angle, avg_angle):
    plt.figure(figsize=(6, 5))
    plt.plot(x, z, marker='o', linestyle='-', color='purple')
    plt.title(f"{file_stem}::{traj_id}\n∑θ={total_angle:.2f} rad, θ̄={avg_angle:.3f} rad/step")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = Path(save_dir) / f"{file_stem}_{traj_id}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📌 已绘制并保存: {save_path}")

def process_and_filter(h5_path, raw_data_dir, turning_dir, plot_dir):
    with h5py.File(h5_path, "r") as f:
        if "follow_paths" not in f:
            return

        file_name = h5_path.name
        file_stem = h5_path.stem
        prefix = "_".join(file_stem.split("_")[:3])
        turned = False

        for traj_id in f["follow_paths"].keys():
            rel_path = f["follow_paths"][traj_id]["rel_path"][:]
            if rel_path.shape[1] < 3:
                continue
            if rel_path.shape[0] == 0:
                continue  # 空轨迹，跳过

            x = rel_path[:, 0]
            z = rel_path[:, 2]
            total_angle, avg_angle = compute_turning_degree(x, z)

            # ✅ 计算首尾连线与 z 轴夹角（单位：度）
            vec_x = x[-1] - x[0]
            vec_z = z[-1] - z[0]
            angle_to_z_rad = np.arctan2(vec_x, vec_z)  # 注意顺序: x 在前，z 在后
            angle_to_z_deg = np.degrees(angle_to_z_rad)

            # ✅ 方向角筛选
            if angle_to_z_deg <= -70 or -25 <= angle_to_z_deg <= 25 or angle_to_z_deg >= 70:
                continue  # 跳过不满足方向性条件的轨迹

            # ✅ 转角筛选
            if total_angle > THRESH_TOTAL_ANGLE and avg_angle > THRESH_AVG_ANGLE:
                print(f"✔ 满足条件: {file_stem}::{traj_id} | ∑θ={total_angle:.2f}, θ̄={avg_angle:.3f}, ∠z={angle_to_z_deg:.1f}°")

                actual_file_stem = f"{prefix}_{traj_id}"
                file_name = actual_file_stem + ".hdf5"
                raw_file = Path(raw_data_dir) / file_name
                dst_file = Path(turning_dir) / file_name

                print(f"raw_file: {raw_file}")
                print(f"dst_file: {dst_file}")
                
                if dst_file.exists():
                    print(f"⏩ 已存在，跳过: {dst_file}")
                    continue
                
                if not dst_file.exists():
                    os.makedirs(turning_dir, exist_ok=True)
                    if raw_file.exists():
                        shutil.copy2(raw_file, dst_file)
                        print(f"📁 已复制到: {dst_file}")
                    else:
                        print(f"⚠ 未找到原文件: {raw_file}")

                plot_single_trajectory(x, z, file_stem, traj_id, plot_dir, total_angle, avg_angle)
                turned = True

        return turned


def batch_scan_and_filter(proc_dir, raw_data_dir, turning_dir, plot_dir):
    proc_dir = Path(proc_dir)
    h5_files = sorted(proc_dir.glob("*.hdf5"))
    for h5_file in h5_files:
        process_and_filter(h5_file, raw_data_dir, turning_dir, plot_dir)

# ==== 示例运行 ====
if __name__ == "__main__":
    proc_dir = "data/proc_data/single_follow"               # 检查轨迹文件位置
    raw_data_dir = "data/split_data/single_follow"          # 原始完整数据
    turning_dir = "data/split_data/turning"                 # 满足转弯的保存路径
    plot_dir = "plots/turning"                              # 图像输出路径
    batch_scan_and_filter(proc_dir, raw_data_dir, turning_dir, plot_dir)
