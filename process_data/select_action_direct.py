import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from numpy.linalg import norm

def is_valid_hdf5(path):
    return h5py.is_hdf5(str(path))

def fit_circle(xs, ys):
    """最小二乘法拟合圆：返回圆心(cx, cy) 和半径r"""
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    b = xs ** 2 + ys ** 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(c + cx**2 + cy**2)
    return cx, cy, r

def classify_by_circle_fit(action):
    """
    基于拟合圆的圆心位置分类：判断圆心在路径朝向左侧还是右侧
    """
    action = np.array(action)
    xs, ys = action[:, 0], action[:, 1]

    if len(xs) < 5:
        return "直行"  # 太短不判断

    try:
        cx, cy, r = fit_circle(xs, ys)
    except Exception as e:
        print(f"❌ 圆拟合失败: {e}")
        return "直行"

    # 初始方向向量
    start_vec = action[1, :2] - action[0, :2]
    start_vec /= norm(start_vec) + 1e-8

    # 连接起点到圆心向量
    to_center_vec = np.array([cx, cy]) - action[0, :2]
    to_center_vec /= norm(to_center_vec) + 1e-8

    # 叉积判断圆心在左/右
    cross = np.cross(start_vec, to_center_vec)

    if cross > 0.5:
        return "左拐"
    elif cross < -0.5:
        return "右拐"
    else:
        return "直行"

def visualize_action(action, label="", idx=None):
    x, y = action[:, 0], action[:, 1]
    plt.plot(x, y, marker='o', label=f"{label}_{idx}", linewidth=1)

def process_one_hdf5(path, global_stats, visualize=False, idx_offset=0):
    if not is_valid_hdf5(path):
        print(f"⚠️  {path.name} 不是有效 HDF5 文件，跳过")
        return 0

    local_count = 0
    try:
        with h5py.File(path, 'r') as f:
            if "follow_paths" not in f:
                return 0

            for key in f["follow_paths"]:
                if "action" not in f["follow_paths"][key]:
                    continue
                action = f["follow_paths"][key]["action"][()]
                label = classify_by_circle_fit(action)
                global_stats[label] += 1
                print(f"[{key}] 分类：{label}")
                if visualize:
                    visualize_action(action, label=label, idx=idx_offset + local_count)
                local_count += 1
    except Exception as e:
        print(f"❌ 打开失败：{path.name}，错误：{e}")

    return local_count

def process_all_hdf5s(folder_path, visualize=True, save_path=None):
    h5_files = sorted(Path(folder_path).glob("*.hdf5"))
    global_stats = Counter()
    total = 0

    for h5_path in h5_files:
        print(f"\n📂 处理文件：{h5_path.name}")
        n = process_one_hdf5(h5_path, global_stats, visualize=visualize, idx_offset=total)
        total += n

    # 输出统计结果
    print("\n📊 全部统计结果：")
    for k in global_stats:
        print(f"{k}: {global_stats[k]} ({global_stats[k]/total:.2%})")

    if visualize:
        plt.title(f"总轨迹分类结果（{total} 条）")
        plt.axis("equal")
        plt.legend(fontsize=7)
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ 图像保存至：{save_path}")
        else:
            plt.show()
        plt.clf()

# === 执行入口 ===
if __name__ == "__main__":
    folder_path = "./data/proc_data/multi_follow"  # 请替换为你的文件夹路径
    output_image = None  # 可设为 None 不保存
    process_all_hdf5s(folder_path, visualize=True, save_path=output_image)
