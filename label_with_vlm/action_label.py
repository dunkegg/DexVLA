import h5py
import numpy as np
import os
from action_utils import (
    extract_data,
    quat_to_yaw,
    yaw_to_angle,
    angle_diff,
    find_future_idx,
    temporal_filter,
)

# hdf5_file = "data/raw_data/rxr_smooth/episode_52.hdf5"


def compute_action_from_yaw(yaw, i, near_idx, far_idx, sharp, slight):
    """
    根据当前 yaw、未来 yaw,判断机器人动作(直走/轻微转向/急转)。
    返回值为: "turn_left", "turn_right", "turn_left_slightly",
               "turn_right_slightly", "go_forward"
    """
    # 当前、较近、较远角度
    # print(
    #     "cur_idx=",
    #     i,
    #     "near_idx=",
    #     near_idx,
    #     "far_idx=",
    #     far_idx
    # )
    cur_angle = yaw_to_angle(yaw[i])
    far_angle = yaw_to_angle(yaw[far_idx])
    near_angle = yaw_to_angle(yaw[near_idx])

    # 角度差（[-pi, pi]）
    near_d_yaw = angle_diff(near_angle, cur_angle)
    # print("near_d_yaw:",near_d_yaw)
    far_d_yaw = angle_diff(far_angle, cur_angle)
    # print("far_d_yaw:",far_d_yaw)

    # 急转（sharp）
    if abs(far_d_yaw) > sharp:
        return "turn_left" if far_d_yaw > 0 else "turn_right"

    # 轻微转向（slight）
    if abs(near_d_yaw) > slight:
        return "turn_left_slightly" if near_d_yaw > 0 else "turn_right_slightly"
    # 默认直走
    return "go_forward"


""" 整个 episode 轨迹 """
# segments = extract_data(hdf5_file)
# follower
# follower_all = segments[0]["follow_pos"]
# print("follower_all:\n", follower_all)
# follower_quat = segments[0]["follow_quat"]
# print("follower_quat:\n", follower_quat)

# 只取第0维和第2维
# follower_xz = follower_all[:, [0, 2]]
# follower_xz[:, 1] = -follower_xz[:, 1]  # z轴数据取相反值
# print("follower_xz:\n",follower_xz)
# print(len(follower_xz))

# yaw_est_follower = quat_to_yaw(follower_quat)
# print("yaw_est_follower:\n",yaw_est_follower)
# print(len(yaw_est_follower))

def get_actions_from_direction_precise(
    follower_xz,
    yaw,
    far_dist=2.0,
    near_dist=0.5,
    stop_thresh=0.02,
    slight_deg=40.0,
    sharp_deg=80.0,
    confirm_k=3,
):
    """
    12.10添加平滑滤波，与动作细致分化,near_idx & far_idx
    """
    N = len(follower_xz)
    slight = np.deg2rad(slight_deg)
    # print("slight:", slight)
    sharp = np.deg2rad(sharp_deg)
    # print("sharp:", sharp)

    raw_actions = []

    for i in range(N):
        # stop 检查
        if i > 0:
            speed = np.linalg.norm(follower_xz[i] - follower_xz[i - 1])
            if speed < stop_thresh:
                raw_actions.append("stop")
                continue
        # ---- 安全获取 far_idx ----
        near_idx = find_future_idx(follower_xz, i, dist_threshold=near_dist)
        far_idx = (
            find_future_idx(follower_xz, i, dist_threshold=far_dist)
            or find_future_idx(follower_xz, i, dist_threshold=near_dist)
            or (N - 1)
        )
        # print("near_idx:", near_idx, "far_idx:", far_idx)

        if near_idx is None:
            raw_actions.append("approaching_final_point")
            continue

        action = compute_action_from_yaw(yaw, i, near_idx, far_idx, sharp, slight)
        raw_actions.append(action)
    # print("raw_actions:\n")
    # for i, a in enumerate(raw_actions[:]):
    #     print(i, a)

    # 简单时序滤波：需要confirm_k个连续相同标注才可以
    final_actions = temporal_filter(raw_actions, confirm_k)
    return final_actions


# actions3 = get_actions_from_direction_precise(
#     follower_xz,
#     yaw_est_follower,
#     near_dist=0.5,
#     far_dist=1.5,
#     slight_deg=25.0,  # slight_deg ≈ 25° ~ 35°
#     sharp_deg=65.0,  # sharp_deg ≈ 55° ~ 70°
# )
# print("aciton3:")
# for i, a in enumerate(actions3[:]):
#     print(i, a)


def save_annotations(hdf5_file, dataset_name, annotations):
    """
    将标注结果写入 HDF5。
    """
    if dataset_name in hdf5_file:
        del hdf5_file[dataset_name]

    cleaned = [s if s is not None else "null" for s in annotations]
    hdf5_file.create_dataset(
        dataset_name, data=np.array(cleaned, dtype=h5py.string_dtype(encoding="utf-8"))
    )
    print(f"💾 已保存数据集: {dataset_name} ({len(annotations)} 条)")


# actions_dataset_name = "annotations_actions0"


# with h5py.File(hdf5_file, "a") as f:  # "a" 表示可写
#     """保存"""
#     save_annotations(f, actions_dataset_name, actions3)


def process_one_episode(file_path, dataset_name="annotations_actions0"):
    """处理单个 episode 文件"""
    print(f"\n📂 Processing: {file_path}")

    try:
        segments = extract_data(file_path)
    except Exception as e:
        print(f"  ❌ Failed to read: {e}")
        return

    follower_all = segments[0]["follow_pos"]
    follower_quat = segments[0]["follow_quat"]

    follower_xz = follower_all[:, [0, 2]]
    follower_xz[:, 1] *= -1

    yaw_est = quat_to_yaw(follower_quat)

    actions = get_actions_from_direction_precise(
        follower_xz,
        yaw_est,
        near_dist=0.5,
        far_dist=1.5,
        slight_deg=25.0,
        sharp_deg=65.0,
    )

    with h5py.File(file_path, "a") as f:
        save_annotations(f, dataset_name, actions)


def process_folder(folder_path):
    """遍历整个文件夹，处理所有 episode_*.hdf5"""
    print(f"🚀 Start batch annotation: folder = {folder_path}")

    files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".hdf5")
    ])

    if not files:
        print("⚠️ No .hdf5 files found.")
        return

    for file_path in files:
        process_one_episode(file_path)

    print("\n🎉 All files processed!")

process_folder("data/raw_data/rxr_smooth/")