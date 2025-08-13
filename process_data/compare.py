import h5py
import numpy as np
from pathlib import Path

def normalize_angle(angle):
    """将任意角度归一化到 [-π, π] 区间"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
def judge_turn_direction_mean(action):
    yaw_start = action[0, 2]
    yaw_end = action[-1, 2]
    yaw_diff = normalize_angle(yaw_end - yaw_start)
    threshold=np.radians(40)
    if yaw_diff > threshold:
        return "right"
    elif yaw_diff < -threshold:
        return "left"
    else:
        return "straight"


    front = action[:5, [0, 1]]
    back = action[-5:, [0, 1]]
    
    mean_front = np.mean(front, axis=0)
    mean_back = np.mean(back, axis=0)

    direction_vec = mean_back - mean_front
    if np.linalg.norm(direction_vec) < 1e-6:
        return "invalid"

    direction_vec = direction_vec / np.linalg.norm(direction_vec)
    reference_vec = np.array([0.0, 1.0])

    cross = reference_vec[0] * direction_vec[1] - reference_vec[1] * direction_vec[0]
    print(cross)
    if cross > 1e-3:
        return "left"
    elif cross < -1e-3:
        return "right"
    else:
        return "straight"

def process_one_file(file_path: Path):
    try:
        with h5py.File(file_path, "r") as f:
            if "action" not in f:
                return "skip"
            action = f["action"][:]
            if action.shape != (30, 3):
                return "skip"
            return judge_turn_direction_mean(action)
    except Exception:
        return "error"

def process_folder(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        print(" 文件夹不存在")
        return

    h5_files = list(folder.glob("*.hdf5"))
    if not h5_files:
        print("文件夹中没有 .hdf5 文件")
        return

    count_left = 0
    count_right = 0
    count_straight = 0
    count_invalid = 0

    print(f"共检测到 {len(h5_files)} 个 HDF5 文件，开始处理...\n")

    for i, f in enumerate(h5_files):
        direction = process_one_file(f)
        if direction == "left":
            count_left += 1
        elif direction == "right":
            count_right += 1
        elif direction == "straight":
            count_straight += 1
        elif direction == "invalid":
            count_invalid += 1
        # 打印每个文件的判断结果
        print(f"{f.name}: {direction}")
        # if i >5:
        #     break

    # 总结统计
    print("\n 统计结果：")
    print(f"左转 : {count_left}")
    print(f"右转 : {count_right}")
    print(f"直行 : {count_straight}")
    print(f"无效 : {count_invalid}")

if __name__ == "__main__":
    folder_path = "data/split_data/turning"  # 替换为你自己的路径
    process_folder(folder_path)
