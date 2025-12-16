import h5py
import numpy as np


def extract_data(hdf5_file):
    """提取hdf5数据"""
    segments = []
    with h5py.File(hdf5_file, "r") as f:
        rel_path = f["rel_path"][:]  # (N,8)
        xyz = rel_path[:, :3]  # 前3个
        # print("xyz:\n",xyz[:10])
        quat = rel_path[:, 3:7]  # 后4个
        # print("quat:\n",quat[:10])
        segments.append(
            {
                "follow_pos": xyz,  # (batch,3,)
                "follow_quat": quat,  # (batch,4,)
            }
        )
    return segments


def quat_to_yaw(q):
    """
    四元数转yaw
    q: array-like [qx, qy, qz, qw]
    returns yaw in radians (range ~ -pi..pi)
    """
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    # yaw around Y axis (XZ plane)
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def yaw_to_angle(y):
    """
    yaw转角度
    """
    vec = np.array([-np.sin(y), np.cos(y)])  # 负号保持你的原始定义
    return np.arctan2(vec[1], vec[0])


def angle_diff(a, b):
    """
    计算两个角度之间最短差值(范围：[-pi, pi])
    """
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def find_future_idx(follower_xz, i, dist_threshold):
    """
    寻找离 follower_xz[i] 欧氏距离 >= dist_threshold 的未来点索引。
    """
    N = len(follower_xz)
    j = i + 1
    while j < N:
        if np.linalg.norm(follower_xz[j] - follower_xz[i]) >= dist_threshold:
            return j
        j += 1
    return None


def temporal_filter(raw_actions, confirm_k):
    """
    时序滤波:连续confirm_k个动作相同时才进行标注,否则沿用之前的动作。
    """
    if confirm_k <= 1:
        return raw_actions

    final_actions = []
    # 滑动确认缓存
    buf = []
    for a in raw_actions:
        buf.append(a)
        if len(buf) < confirm_k:
            final_actions.append(buf[-1])  # provisional
        else:
            # 检查最后 confirm_k 个动作
            if all(x == buf[-1] for x in buf[-confirm_k:]):
                final_actions.append(buf[-1])
            else:
                # keep previous final (or provisional)
                final_actions.append(final_actions[-1] if final_actions else buf[-1])
        # keep buffer small
        if len(buf) > confirm_k * 3:
            buf = buf[-confirm_k * 2 :]
    return final_actions
