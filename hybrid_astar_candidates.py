import numpy as np
from typing import List
from human_follower.hybrid_a.planner import HybridAStar      # 你的 HybridAStar 类
import magnum as mn
def _resample_to_n(points, n=10):
    if points is None or len(points) == 0:
        return None
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    t = np.linspace(0, len(points) - 1, n)
    idx = np.arange(len(points))
    out = np.stack([np.interp(t, idx, points[:, i]) for i in range(3)], axis=1)
    return out.astype(np.float32)


def make_k_hybrid_astar_paths(sim,
                              start_xyz,
                              goal_xyz,
                              start_yaw,
                              goal_yaw,
                              k=7,
                              xy_threshold=0.5,
                              yaw_threshold=np.deg2rad(10),
                              height=0.0):
    """
    注意：start_xyz / goal_xyz / start_yaw / goal_yaw 均为 *局部坐标系* 下的量.
    """
    start_xyz = np.asarray(start_xyz, dtype=np.float32)
    goal_xyz  = np.asarray(goal_xyz , dtype=np.float32)

    y_h = height

    paths = []
    # heuristic_list    = [3.0, 3.5, 4.0, 4.5]    # 不同的启发式权重
    heuristic_list    =  [1.1, 1.4, 1.7, 2.0]   # 更顺滑
    # heuristic_list    =   [4.5, 4.8, 5.1, 5.4]  # 花苞状
    steering_num_list = [10]              # 不同的转向档数
# 
    for h_w in heuristic_list:
        for n_steer in steering_num_list:
            if len(paths) >= k:
                break

            planner = HybridAStar(
                sim=sim,
                heuristic_weight=h_w,
                xy_threshold=0.2,
                yaw_threshold=yaw_threshold,
                step_length=0.15,
                rear_wheelbase=0.1,
                height=y_h
            )

            # 自定义阈值
            xy_thr  = xy_threshold
            yaw_thr = yaw_threshold

            planner.xy_threshold  = xy_thr
            planner.yaw_threshold = yaw_thr

            # 修改转向档数
            R = 0.1  # 期望最小转弯半径 越小越灵活 # 在这里改 
            delta_max = np.arctan(planner.L / R)
            planner.steers = np.linspace(-delta_max, delta_max, n_steer)

            path2d = planner.plan(
                (start_xyz[0], start_xyz[2], start_yaw),
                (goal_xyz[0],  goal_xyz[2],  goal_yaw)
            )

            if path2d is None:
                continue

            xyz = np.array([[p[0], y_h, p[1]] for p in path2d], dtype=np.float32)
            p10 = _resample_to_n(xyz, n=10)
            paths.append(p10)

    return paths[:k]

