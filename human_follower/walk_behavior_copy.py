#walk_behavior_copy.py
import os
import gzip
import json
import yaml
import numpy as np
import habitat_sim
import math
import sys
import magnum as mn
from tqdm import tqdm
from PIL import Image
import imageio
import random
import logging
import time
from human_follower.hybrid_a.planner import HybridAStar
import matplotlib.pyplot as plt
from habitat_sim.utils.common import quat_from_coeffs, quat_from_two_vectors , quat_from_angle_axis, quat_to_angle_axis # 把 [w,x,y,z] 转 Quaternion
from human_follower.save_data import to_quat
from habitat_for_sim.agent.path_generator import generate_path
from habitat_for_sim.utils.frontier_exploration import FrontierExploration

from habitat_for_sim.utils.goat import calculate_euclidean_distance
from hybrid_astar_candidates import make_k_hybrid_astar_paths
from process_data.process_raw_h5 import world2local

def to_vec3(v) -> mn.Vector3:
    """接受 magnum.Vector3 或 list/tuple/np.ndarray"""
    if isinstance(v, mn.Vector3):
        return v
    return mn.Vector3(float(v[0]), float(v[1]), float(v[2]))

def quat_from_two_vectors_mn(v0, v1, eps=1e-8):
    a = to_vec3(v0).normalized()
    b = to_vec3(v1).normalized()
    c = mn.math.clamp(mn.math.dot(a, b), -1.0, 1.0)

    if c < -1.0 + eps:                       # 反向
        orth = mn.Vector3(1,0,0) if abs(a.x) < 0.9 else mn.Vector3(0,1,0)
        axis = mn.math.cross(a, orth).normalized()
        return mn.Quaternion.rotation(mn.Rad(math.pi), axis)

    axis  = mn.math.cross(a, b)
    s     = math.sqrt((1.0 + c) * 2.0)
    inv_s = 1.0 / s
    # 这里用 (vector, scalar) 构造
    q = mn.Quaternion(
            mn.Vector3(axis.x * inv_s,
                       axis.y * inv_s,
                       axis.z * inv_s),
            s * 0.5
        ).normalized()
    return q     


def shortest_angle_diff(a, b):
    """
    返回 b−a 的最短角差，范围 (-π, π]
    """
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return diff


def simulate(sim, dt, get_observations=False):
    r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
    observations = []
    target_time = sim.get_world_time() + dt
    while sim.get_world_time() < target_time:
        sim.step_physics(0.1 / 60.0)
        if get_observations:
            observations.append(sim.get_sensor_observations())
    return observations

def _to_xyz_pos(pos):  # <<< NEW
    """mn.Vector3 或 array-like -> np.float32[3]"""
    if isinstance(pos, mn.Vector3):
        return np.array([pos.x, pos.y, pos.z], dtype=np.float32)
    return np.asarray(pos, dtype=np.float32).reshape(3)


# def clip_by_distance2target(path, distance, target_pos=None):
#     if target_pos is None:
#         target_pos = np.array(path[-1][0]) 
#      # 获取目标点的位置
#     clipped_path = [
#         (pos, quat, yaw) for pos, quat, yaw in path
#         if np.linalg.norm(np.array(pos) - target_pos) > distance  # wzj
#     ]

#     return clipped_path

def clip_by_distance2target(path, distance, target_pos=None, min_len=2):  # <<< CHANGED
    """
    path: List[(pos, quat, yaw)]
    仅保留到目标点距离 > distance 的片段；若过短则回退保留尾部 min_len 个元素。
    """
    if not path:
        return []

    if target_pos is None:
        target_pos = _to_xyz_pos(path[-1][0])
    else:
        target_pos = _to_xyz_pos(target_pos)

    kept = []
    for (pos, quat, yaw) in path:
        p = _to_xyz_pos(pos)
        if np.linalg.norm(p - target_pos) > float(distance):
            kept.append((pos, quat, yaw))

    if len(kept) < min_len:
        return path[-min_len:] if len(path) >= min_len else path[:]

    return kept

# def get_path_with_time(raw_path, time_step = 0.1, speed = 0.5):
#     #raw path : postion, quat, yaw
def slerp_quaternion(q1: mn.Quaternion, q2: mn.Quaternion, t: float) -> mn.Quaternion:
    """
    对两个 Magnum 四元数执行球面线性插值（SLERP）
    保证数值稳定性，避免 NaN
    """
    q1 = q1.normalized()
    q2 = q2.normalized()

    dot = mn.math.dot(q1.vector, q2.vector)

    if dot < 0.0:
        q2 = mn.Quaternion(-q2.vector)
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # 太接近，使用线性插值并归一化
        interp = (q1.vector * (1 - t) + q2.vector * t).normalized()
        return mn.Quaternion(interp)

    theta_0 = math.acos(dot)
    sin_theta = math.sin(theta_0)

    w1 = math.sin((1 - t) * theta_0) / sin_theta
    w2 = math.sin(t * theta_0) / sin_theta

    interp_vector = q1.vector * w1 + q2.vector * w2
    return mn.Quaternion(interp_vector.normalized())

def get_path_with_time(raw_path, time_step=0.1, speed=0.5):
    """
    对原始路径按速度插值，生成细化的 (position, quat, yaw) 路径

    Args:
        raw_path: List of (position: mn.Vector3, quat: mn.Quaternion, yaw: float) tuples
        time_step: float, 插值时间间隔（单位：秒）
        speed: float, 行进速度（单位：米/秒）

    Returns:
        new_path: List of (position: mn.Vector3, quat: mn.Quaternion, yaw: float) 插值后的路径
    """
    new_path = []
    if len(raw_path) < 2:
        return new_path

    step_dist = speed * time_step

    for i in range(len(raw_path) - 1):
        start_pos, start_quat, start_yaw = raw_path[i]
        end_pos, end_quat, end_yaw = raw_path[i + 1]

        seg_vec = end_pos - start_pos
        seg_len = seg_vec.length()
        
        if seg_len < 1e-4:
            continue

        direction = seg_vec.normalized()
        yaw_diff = shortest_angle_diff(start_yaw, end_yaw)

        n_steps = int(np.ceil(seg_len / step_dist))
        for step in range(n_steps):
            frac = (step * step_dist) / seg_len

            interp_pos = start_pos + direction * (frac * seg_len)
            interp_yaw = start_yaw + yaw_diff * frac
            interp_quat = slerp_quaternion(start_quat, end_quat, frac)

            new_path.append((interp_pos, interp_quat, interp_yaw))

    # 确保最后一个点包含在内
    last_pos, last_quat, last_yaw = raw_path[-1]
    new_path.append((last_pos, last_quat, last_yaw))

    return new_path


def generate_interfere_path_from_target_path(follow_path, agent, time_step=0.1, speed=0.5, radius=1.0):
    """
    根据目标人的轨迹生成干扰人的扰动路径
    Args:
        target_path: 原始路径 [(pos, quat, yaw)]
        agent: AgentHumanoid 实例（用于初始位置）
        time_step: 时间分辨率
        speed: 插值速度
        radius: 生成扰动点的偏移半径

    Returns:
        List of (pos, yaw) 干扰人路径
    """
    # dense_path = get_path_with_time(follow_path, time_step, speed)

    interfere_path = []
    distance = 0
    start_pos = follow_path[0][0]
    threshold= 0.5
    for pos, quat, yaw in follow_path:
        distance += calculate_euclidean_distance([start_pos.x, start_pos.y,start_pos.z],[pos.x, pos.y,pos.z])
        start_pos = pos
        if distance > threshold:
            distance = 0
            # 在主路径每个点附近生成一个扰动点
            offset = mn.Vector3(
                random.uniform(-radius, radius),
                0,
                random.uniform(-radius, radius)
            )
            new_pos = pos + offset
            interfere_path.append((new_pos, quat, yaw))

    dense_path = get_path_with_time(interfere_path, time_step, speed)
    return dense_path

def generate_interfere_sample_from_target_path(follow_path, pathfinder, radius=1.0):
    """
    沿主路径采样点生成干扰人轨迹，每个点在正前方向一定角度范围内偏移
    """
    interfere_path = []
    sample_distance = 0
    total_distance = 0
    threshold = 0.5
    start_pos = follow_path[0][0]
    circle_random = random.choice([True, False])
    for i in range(1, len(follow_path)):
        pos, _, yaw = follow_path[i]
        dis_diff = calculate_euclidean_distance(
            [start_pos.x, start_pos.y, start_pos.z],
            [pos.x, pos.y, pos.z]
        )
        sample_distance += dis_diff
        total_distance += dis_diff
        start_pos = pos

        if sample_distance < threshold:
            continue
        sample_distance = 0

        # 增大随机幅度
        weight_radius = math.log(total_distance + 1) * radius

        
        if circle_random:
            for _ in range(10):
                offset = mn.Vector3(
                    random.uniform(-weight_radius, weight_radius),
                    0,
                    random.uniform(-weight_radius, weight_radius)
                )
                new_pos = pos + offset
                real_coords = np.array([new_pos.x, new_pos.y, new_pos.z])
                if pathfinder.is_navigable(real_coords):
                    interfere_path.append(new_pos)
                    break
        else:
            # 方向向量（从 i-1 到 i）
            prev_pos = follow_path[i - 1][0]
            forward_vec = (pos - prev_pos).normalized()

            # 生成前方 ±60° 扇形内随机扰动向量
            max_angle = math.radians(60)
            for _ in range(10):
                angle = random.uniform(-max_angle, max_angle)
                rot_mat = mn.Matrix4.rotation_y(mn.Rad(angle))
                new_dir = rot_mat.transform_vector(forward_vec)
                offset = new_dir * random.uniform(0.1, weight_radius)

                new_pos = pos + offset
                real_coords = np.array([new_pos.x, new_pos.y, new_pos.z])
                if pathfinder.is_navigable(real_coords):
                    interfere_path.append(new_pos)
                    break

    return interfere_path

def generate_interfer_path(interfering_humanoids, human_path, time_step=1/10, speed=0.7, radius=1.5):
    """
    为所有干扰人生成扰动轨迹，并初始化路径状态

    Args:
        interfering_humanoids (list): AgentHumanoid 实例列表
        human_path (list): 目标人的轨迹 [(pos, quat, yaw)]
        time_step (float): 插值时间步长
        speed (float): 干扰人移动速度
        radius (float): 偏移扰动半径
    """
    for interferer in interfering_humanoids:
        path = generate_interfere_path_from_target_path(
            human_path, interferer,
            time_step=time_step, speed=speed, radius=radius
        )
        interferer.reset_path(path)

def _tuple_path_to_xyz_np(path):
    """
    将 [(pos, quat, yaw), ...] 或 [(Vector3,...), ...] 转成 [N,3] np.ndarray
    """
    if path is None or len(path) == 0:
        return None
    coords = []
    for p in path:
        if isinstance(p, tuple) or isinstance(p, list):
            pos = p[0]  # 第一个元素是位置
        else:
            pos = p
        if isinstance(pos, mn.Vector3):
            coords.append([pos.x, pos.y, pos.z])
        else:
            coords.append(list(pos))
    return np.array(coords, dtype=np.float32)

# def generate_fan_paths_local(
#         sim,
#         start_xyz_local: np.ndarray,
#         start_yaw_local: float,
#         real_goal_xyz_local: np.ndarray,
#         real_goal_yaw_local: float,
#         num_endpoints: int = 5,
#         fan_angle_deg: float = 90.0,
#         k_per_goal: int = 5,
#         xy_threshold: float = 0.5,
#         yaw_threshold: float = math.radians(10)
#     ):
#     """
#     在局部坐标系中以起点为圆心、真实终点距离为半径，按照扇形方向
#     (± fan_angle_deg/2) 生成 num_endpoints 个目标点 ，每个目标点跑 K 条
#     Hybrid A* (make_k_hybrid_astar_paths)。返回所有路径的list.
#     """
#     # 扇形角度
#     half_angle = math.radians(fan_angle_deg) / 2.0
#     # 从真实目标向量
#     vec = real_goal_xyz_local - start_xyz_local
#     radius = np.linalg.norm(vec)

#     # 真实方向
#     base_angle = math.atan2(vec[2], vec[0])  # (x,z) -> angle

#     # 生成采样角
#     if num_endpoints > 1:
#         angles = np.linspace(base_angle - half_angle,
#                              base_angle + half_angle,
#                              num_endpoints)
#     else:
#         angles = np.array([base_angle])

#     all_paths = []

#     for ang in angles:
#         # 扇形目标位置
#         gx = radius * math.cos(ang)
#         gz = radius * math.sin(ang)
#         goal_xyz = np.array([gx, 0.0, gz], dtype=np.float32)
#         # Yaw 使用真实终点 yaw
#         goal_yaw = real_goal_yaw_local

#         # 调用已有的函数
#         paths_k = make_k_hybrid_astar_paths(
#             sim         = sim,
#             start_xyz   = start_xyz_local,
#             goal_xyz    = goal_xyz,
#             start_yaw   = start_yaw_local,
#             goal_yaw    = goal_yaw,
#             k           = k_per_goal,
#             xy_threshold= xy_threshold,
#             yaw_threshold = yaw_threshold
#         )
#         all_paths.extend(paths_k)

#     return all_paths

# def generate_fan_paths_local(
#         sim,
#         start_xyz_local: np.ndarray,
#         start_yaw_local: float,
#         real_goal_xyz_local: np.ndarray,
#         real_goal_yaw_local: float,
#         num_endpoints: int = 5,
#         fan_angle_deg: float = 90.0,
#         num_yaws: int = 3,
#         yaw_range_deg: float = 60.0,
#         k_per_goal: int = 5,
#         xy_threshold: float = 0.5,
#         yaw_threshold: float = math.radians(10)
#     ):
#     """
#     在局部坐标系中构造 “扇形位置 × yaw散射” 的目标集合，每个目标调用HybridA*。
#     """
#     # 扇形角度
#     half_angle = math.radians(fan_angle_deg) / 2.0
#     # 从真实目标向量
#     vec = real_goal_xyz_local - start_xyz_local
#     radius = np.linalg.norm(vec)
#     base_angle = math.atan2(vec[2], vec[0])  # (x,z) -> angle

#     # 扇形 sampling
#     if num_endpoints > 1:
#         pos_angles = np.linspace(base_angle - half_angle,
#                                  base_angle + half_angle,
#                                  num_endpoints)
#     else:
#         pos_angles = np.array([base_angle])

#     # yaw sampling
#     half_yaw = math.radians(yaw_range_deg) / 2.0
#     if num_yaws > 1:
#         yaws = np.linspace(real_goal_yaw_local - half_yaw,
#                            real_goal_yaw_local + half_yaw,
#                            num_yaws)
#     else:
#         yaws = np.array([real_goal_yaw_local])

#     all_paths = []

#     # -- 对每个位置 & 每个yaw组合做一次 A*
#     for ang in pos_angles:
#         gx = radius * math.cos(ang)
#         gz = radius * math.sin(ang)
#         goal_xyz = np.array([gx, 0.0, gz], dtype=np.float32)

#         for goal_yaw in yaws:
#             paths_k = make_k_hybrid_astar_paths(
#                 sim         = sim,
#                 start_xyz   = start_xyz_local,
#                 goal_xyz    = goal_xyz,
#                 start_yaw   = start_yaw_local,
#                 goal_yaw    = goal_yaw,
#                 k           = k_per_goal,
#                 xy_threshold= xy_threshold,
#                 yaw_threshold = yaw_threshold
#             )
#             all_paths.extend(paths_k)

#     return all_paths

def generate_fan_paths_local(
        sim,
        start_xyz_local: np.ndarray,
        start_yaw_local: float,
        real_goal_xyz_local: np.ndarray,
        real_goal_yaw_local: float,
        num_endpoints: int = 5,
        fan_angle_deg: float = 90.0,
        num_yaws: int = 3,
        yaw_range_deg: float = 60.0,
        k_per_goal: int = 5,
        xy_threshold: float = 0.5,
        yaw_threshold: float = math.radians(10)
    ):
    """
    在局部坐标系中构造 “扇形位置 × yaw散射” 的目标集合，每个目标调用HybridA*。
    yaw 散射的中心是“起点-->该虚拟终点”的方向，而不是真实目标的 yaw。
    """

    # 扇形角 (half_angle)
    half_angle = math.radians(fan_angle_deg) / 2.0

    # 起点->真实终点向量长度作为半径
    vec = real_goal_xyz_local - start_xyz_local
    radius = np.linalg.norm(vec)

    # 起点→真实终点的方向角 (作为扇形中心方向)
    base_angle = math.atan2(vec[2], vec[0])  # (x,z)->angle

    # 生成扇形内的位置角
    if num_endpoints > 1:
        pos_angles = np.linspace(base_angle - half_angle,
                                 base_angle + half_angle,
                                 num_endpoints)
    else:
        pos_angles = np.array([base_angle])

    # yaw 散射范围
    half_yaw = math.radians(yaw_range_deg) / 2.0

    all_paths = []

    for ang in pos_angles:
        # 虚拟终点的位置
        gx = radius * math.cos(ang)
        gz = radius * math.sin(ang)
        goal_xyz = np.array([gx, 0.0, gz], dtype=np.float32)

        # <=== 关键：用“起点→该虚拟终点”的方向 dir_yaw 作为 yaw 的中心 ===>
        dir_yaw = math.atan2(gz - start_xyz_local[2],
                             gx - start_xyz_local[0])

        # around dir_yaw ±half_yaw
        if num_yaws > 1:
            yaws = np.linspace(dir_yaw - half_yaw,
                               dir_yaw + half_yaw,
                               num_yaws)
        else:
            yaws = np.array([dir_yaw])

        # 对每个 (目标位置, yaw) 组合调用 HybridA*
        for goal_yaw in yaws:
            paths_k = make_k_hybrid_astar_paths(
                sim         = sim,
                start_xyz   = start_xyz_local,
                goal_xyz    = goal_xyz,
                start_yaw   = start_yaw_local,
                goal_yaw    = goal_yaw,
                k           = k_per_goal,
                xy_threshold= xy_threshold,
                yaw_threshold = yaw_threshold
            )
            all_paths.extend(paths_k)

    return all_paths


def sample_data(sim, origin_follow_path, goal_pos, follow_state,
                human_state, cur_obs=None, goal_yaw=None, follow_yaw=None,
                pre_cands=None, episode_idx=None, step_idx=None):
    """
    输入：
      - origin_follow_path: 主路径片段
      - follow_state: 当前agent位姿 (world frame)
      - human_state: 当前目标人物位置 (world frame [x,y,z])
      - goal_pos, goal_yaw: 世界坐标下的目标位置和yaw
    返回：
      - feasible_idx, gt_10, cands_10
    """
    
    keep_distance = 0.7

    def _resample_to_10(xyz_np, n=10):
        if xyz_np is None or len(xyz_np) == 0:
            return None
        if len(xyz_np) == n:
            return xyz_np.astype(np.float32)
        t = np.linspace(0, len(xyz_np) - 1, n)
        idx = np.arange(len(xyz_np))
        out = np.stack([np.interp(t, idx, xyz_np[:, i]) for i in range(3)], axis=1)
        return out.astype(np.float32)

    def _match_feasible_idx(cands_10, gt_10):
        dists = [np.linalg.norm(c - gt_10, axis=1).mean() for c in cands_10]
        arg = int(np.argmin(dists))
        return arg, float(dists[arg])

    # ------------------------------------------------------------------
    # >> ①把 world → local   (坐标 & yaw)
    wxyz = follow_state.rotation
    quat_mn = mn.Quaternion(mn.Vector3(wxyz.x, wxyz.y, wxyz.z), wxyz.w)
    
   
    # rel_path shape = (1,8)
    rel = np.zeros((1,8), dtype=np.float32)
    rel[0, :3] = (goal_pos - follow_state.position)
    rel[0, 7]  = goal_yaw

    local_path, _ = world2local(
        rel_path    = rel,
        human_state = goal_pos,
        follow_quat = quat_mn,
        follow_yaw  = follow_yaw,
        type        = 1
    )

    # 局部值
    start_xyz_local = np.zeros(3, dtype=np.float32)
    goal_xyz_local  = local_path[0, :3]
    start_yaw_local = math.pi/2
    goal_yaw_local  = float(local_path[0,7]+math.pi/2)
    # ------------------------------------------------------------------

    # # >> ② 调用混合A*（使用局部起点 / 局部终点 / yaw）
    # paths_10 = make_k_hybrid_astar_paths(
    #     sim           = sim,
    #     start_xyz     = start_xyz_local,
    #     goal_xyz      = goal_xyz_local,
    #     start_yaw     = start_yaw_local,
    #     goal_yaw      = goal_yaw_local,
    #     k             = 5,
    #     xy_threshold  = 0.5,
    #     yaw_threshold = math.radians(10),
    # )
    t0 = time.time()
    
    paths_10 = generate_fan_paths_local(
        sim,
        start_xyz_local,
        start_yaw_local,
        goal_xyz_local,
        goal_yaw_local,
        num_endpoints = 19,
        fan_angle_deg = 360.0,
        num_yaws      = 2,      # <- 新增：每个位置再采3个yaw
        yaw_range_deg = 50.0,   # <- yaw 发散范围 ±30°
        k_per_goal    = 4,  # 每个目标生成多少条 Hybrid A*
        xy_threshold  = 1,
        yaw_threshold = math.radians(15),
    )

    t1 = time.time()
    print(f"Hybrid A* 生成 {len(paths_10)} 条候选路径，耗时 {t1-t0:.2f} 秒")

    # # ============== 可视化 (当前 observation + 局部候选路径)  ==================
    # if episode_idx is not None and step_idx is not None:
    #     save_path = f"plots/a_star/episode_{episode_idx}_t{step_idx}.png"
    #     import matplotlib.pyplot as plt
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    #     # 左侧画 RGB
    #     if cur_obs is not None:
    #         axs[0].imshow(cur_obs)
    #         axs[0].set_title("current obs")
    #         axs[0].axis("off")

    #     # 右侧画局部 Hybrid A* 路径
    #     for i, p in enumerate(paths_10):
    #         p = np.asarray(p, dtype=np.float32)
    #         axs[1].plot(p[:,0], p[:,2], lw=1.8, label=f"cand {i}")
    #     axs[1].scatter(0.0, 0.0, c='red', s=50, marker='*', label='start (local)')
    #     axs[1].scatter(goal_xyz_local[0], goal_xyz_local[2],
    #                 c='green', s=50, marker='*', label='goal (local)')
    #     axs[1].set_aspect("equal")
    #     axs[1].set_title("Hybrid A* candidates (local)")
    #     axs[1].set_xlabel("X (lateral)")  # 添加X轴标签（横向）
    #     axs[1].set_ylabel("Z (longitudinal)")  # 添加Z轴标签（纵向）
    #     axs[1].legend(fontsize="x-small")
    #     axs[1].grid(True, linestyle='--', alpha=0.5)  # 可选：添加网格线，提高可读性

    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.tight_layout()
    #     fig.savefig(save_path, dpi=150)
    #     plt.close(fig)

    # --- 可视化 (local)
    if episode_idx is not None and step_idx is not None:
        save_path = f"plots/a_star_1/episode_{episode_idx}_t{step_idx}.png"
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # 左侧画 RGB
        if cur_obs is not None:
            axs[0].imshow(cur_obs)
            axs[0].set_title("current obs")
            axs[0].axis("off")
        # 右侧画局部 Hybrid A* 路径
        for p in paths_10:
            p = np.asarray(p)
            axs[1].plot(p[:,0], p[:,2])
        # fig,ax = plt.subplots(1,1,figsize=(5,5))
        axs[1].scatter(0,0,c='red',label='start')
        axs[1].scatter(goal_xyz_local[0],goal_xyz_local[2],c='green',label='real goal')
        axs[1].set_aspect("equal"); axs[1].legend()
        plt.tight_layout();  os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path,dpi=120); plt.close()


    if paths_10 is None or len(paths_10) == 0:
        return None, None, None

    # 2) 生成 GT 片段（基于当前窗口，裁剪到 keep_distance，并重采样）
    # ----- GT 片段（从 origin_follow_path 保距裁剪 -> 统一10点） -----
    gt_segment = clip_by_distance2target(origin_follow_path, keep_distance)
    gt_xyz_np = _tuple_path_to_xyz_np(gt_segment)
    if gt_xyz_np is None or len(gt_xyz_np) < 2:
        return None, None, None

    gt_10 = _resample_to_10(gt_xyz_np, n=10)     # [10,3]
    feasible_idx, _ = _match_feasible_idx(paths_10, gt_10)

    return feasible_idx, gt_10.tolist(), [p.tolist() for p in paths_10]


def walk_along_path_multi(
    all_index,
    sim,
    humanoid_agent,  # AgentHumanoid 实例
    human_path,
    fps=10,
    timestep_gap = 0.2,
    forward_speed=0.7,
    interfering_humanoids=None,
    robot = None,
    pre_cands=None,
):
    if robot is not None:
        robot.set_episode_id(all_index)
    # output = {"obs": [], "follow_paths": []}
    output = {"obs": [], "follow_groups": []}

    follow_success = False

    keep_distance = 0.7


    observations = []
    humanoid_agent.step_directly(
        target_pos=human_path[0][0],
        target_yaw=human_path[0][2],
    )
    sim.step_physics(1.0 / fps)

    follow_state = sim.agents[0].get_state()
    follow_yaw = human_path[0][2]
    follow_state.position = human_path[0][0]
    follow_state.rotation = to_quat(human_path[0][1])
    sim.agents[0].set_state(follow_state)
    obs = sim.get_sensor_observations(0)


    now = 0
    if robot is not None:
        robot.set_obs(obs['color_0_0'], now, save=True)

    follow_timestep = 0

    move_dis = 0
    last_dis = 0
    last_plan_dis = 0

    last_sample_time = 0
    last_plan_time = 0
    last_step_time = 0
    
    for time_step in range(2, len(human_path)):
        goal_pos, goal_quat, goal_yaw = human_path[time_step]
        

        # 获取 humanoid 当前状态
        start_pos, start_yaw, _ = humanoid_agent.get_pose()

        seg_vec = goal_pos - start_pos
        seg_len = seg_vec.length()
        move_dis += seg_len
        if seg_len < 1e-4:
            continue

        direction = seg_vec.normalized()

        # 调用控制器移动一步
        humanoid_agent.step_with_controller(
            target_pos=goal_pos,
            target_yaw=goal_yaw,
            direction=direction,
        )


        # ▶ 插入干扰人形位置 todo
        if interfering_humanoids:
            # for interferer in interfering_humanoids:
            #     interferer.place_near_goal(goal_pos, radius=2.0)
        
            for interferer in interfering_humanoids:
                if hasattr(interferer, "interfere_path") and time_step < len(interferer.interfere_path):
                    pos, quat,yaw = interferer.interfere_path[time_step]
                    if time_step>0:
                        interferer_direction = (interferer.interfere_path[time_step][0] - interferer.interfere_path[time_step-1][0]).normalized()
                    else:
                        interferer_direction = direction
                    interferer.step_with_controller(pos, yaw, interferer_direction)
                    interferer.time_step += 1

        # 更新物理引擎
        sim.step_physics(1.0 / fps)
        now = timestep_gap * time_step
        obs = sim.get_sensor_observations(0)
        observations.append(obs)
        # if robot is None:
        #     sample_gap = 1.0
        #     if now - last_sample_time < sample_gap:
        #         continue
        #     last_sample_time = now
            
        #     follow_timestep += 1
        #     follow_state.position = human_path[follow_timestep][0]
        #     follow_state.rotation = to_quat(human_path[follow_timestep][1])
        #     follow_yaw = human_path[follow_timestep][2]
        #     sim.agents[0].set_state(follow_state)
        #     origin_follow_path = human_path[follow_timestep:time_step]
        #     feasible_idx, gt_10 = sample_data(sim=sim, origin_follow_path = origin_follow_path,
        #                                                                     goal_pos = goal_pos,
        #                                                                     follow_state = follow_state)
        #     follow_timestep += int(sample_gap/timestep_gap)
        #     # 5) 落盘到 output["follow_groups"]
        #     if "follow_groups" not in output:
        #         output["follow_groups"] = []
        #     if feasible_idx:
        #         group = {
        #             "obs_idx": len(observations)-1,
        #             "follow_state": human_path[follow_timestep],
        #             "human_state": human_path[time_step],
        #             # "cands": np.stack(paths_10, 0).tolist(),   # [K,10,3] -> list 存储更通用
        #             "feasible_idx": int(feasible_idx),
        #             "tau_gt": gt_10.tolist(),                  # 直接把 GT 的 10 路标也存下来
        #             "desc": humanoid_agent.get_desc(),
        #             "type": int(0),
        #         }
        #         output["follow_groups"].append(group)

        if robot is None:
            sample_gap = 1.0  # <<< CHANGED: 用 float，单位同 now（秒）
            if now - last_sample_time < sample_gap:
                continue
            last_sample_time = now

            # 推进一步（防止越界）
            next_follow_timestep = min(len(human_path) - 1, follow_timestep + 1)  # <<< NEW
            follow_timestep = next_follow_timestep
            if follow_timestep > time_step - 25:
                continue

            # 更新跟随者位姿
            follow_state.position = human_path[follow_timestep][0]
            follow_state.rotation = to_quat(human_path[follow_timestep][1])
            follow_yaw = human_path[follow_timestep][2]
            sim.agents[0].set_state(follow_state)
            start_yaw_world, _ = quat_to_angle_axis(follow_state.rotation)
            start_yaw_world =  -start_yaw + math.pi/2


            # 当前 GT 窗口：从 follow_timestep 到 time_step（含）
            origin_follow_path = human_path[follow_timestep:time_step]
            feasible_idx, gt_10, paths_10 = sample_data(
                sim=sim,
                origin_follow_path=origin_follow_path,
                goal_pos=goal_pos,
                follow_state=follow_state,       
                human_state=human_path[time_step],  
                cur_obs = observations[-1]['color_0_0'],  # <<< ADD: 当前观测
                goal_yaw=goal_yaw,
                pre_cands=pre_cands, 
                follow_yaw=follow_yaw,
                episode_idx=all_index,
                step_idx=time_step,
            )

            # 保持采样步进（按秒换算成帧数）
            step_frames = max(1, int(sample_gap / max(1e-6, timestep_gap)))  # <<< CHANGED
            follow_timestep = min(len(human_path) - 1, follow_timestep + step_frames)  # <<< CHANGED

            # 落盘 follow_groups
            if "follow_groups" not in output:
                output["follow_groups"] = []

            # 注意：feasible_idx 可能是 0，不能用 if feasible_idx 判断  # <<< CHANGED
            if feasible_idx is not None and gt_10 is not None:  # <<< CHANGE
                group = {
                    "obs_idx": len(observations)-1,
                    "follow_state": human_path[min(follow_timestep, len(human_path)-1)],
                    "human_state": human_path[time_step],
                    "feasible_idx": int(feasible_idx),
                    "tau_gt": gt_10,                           # [10,3]
                    "cands": paths_10 if paths_10 is not None else [],  # <<< ADD
                    "desc": humanoid_agent.get_desc(),
                    "type": int(0),
                }
                output["follow_groups"].append(group)


        else:
            
            sample_fps = 1.3
            sample_fps = 3
            plan_fps = 10
            follow_size = 5
            output["sample_fps"] = sample_fps
            output["plan_fps"] = plan_fps
            output["follow_size"] = follow_size
            # step_fps = 0.4
            
            if now - last_sample_time >= 1/sample_fps:
                last_sample_time = now
                obs = sim.get_sensor_observations(0)
                robot.set_obs(obs['color_0_0'], now, save=True)
            if now - last_plan_time >= 1/plan_fps and now > 3:
                last_plan_time = now
                robot.eval_bc()
                position,yaw,quat = humanoid_agent.get_pose()
                robot.save_obs(now, position)
                robot.compare_step(follow_size, forward_speed/plan_fps)

            if time_step == len(human_path)-1:
                step_range = int(1.5/(forward_speed*timestep_gap))
                last_sample = 0
                for i in range(step_range):
                    if (i/forward_speed)*timestep_gap - last_sample >= 1/sample_fps:
                        last_sample = (i/forward_speed)*timestep_gap
                        obs = sim.get_sensor_observations(0)
                        robot.set_obs(obs['color_0_0'], now, save=True)
                    robot.eval_bc()
                    position,yaw,quat = humanoid_agent.get_pose()
                    robot.save_obs(now, position)
                    robot.compare_step(follow_size, forward_speed/plan_fps)
                
                if calculate_euclidean_distance(robot.get_state().position, goal_pos) < 2:
                    follow_success = True


    if robot:
        observations = [{"color_0_0": obs} for obs in robot.get_observations()]
        output["follow_result"] = follow_success
    output["obs"] = observations

    print("walk done")
    return output

