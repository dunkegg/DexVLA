import numpy as np
import matplotlib.pyplot as plt
import magnum as mn
import math

def wrap_pi(angle):
    """把任意角度包到 (-π, π] 区间"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def world2local(rel_path: np.ndarray,
                human_state:  np.ndarray,
                follow_quat: mn.Quaternion,
                follow_yaw: float,
                type: int) -> np.ndarray:
    """
    rel_path : (N, 8)  已经做过 位置减 follow_pos
               列序 [x y z w x y z yaw_world]
    follow_quat : mn.Quaternion  (w, xyz)  用于旋转向量
    follow_yaw  : float (rad)    已有的朝向标量
    """
    R_inv = follow_quat.inverted()          # q_f⁻¹
    out = rel_path.copy()

    human_local = R_inv.transform_vector(mn.Vector3(*human_state))
    # 1) 位置向量旋转到局部系
    for i, v in enumerate(rel_path[:, :3]):
        v_local = R_inv.transform_vector(mn.Vector3(*v))
        if type ==1:
            # out[i, :3] = [-v_local.x, v_local.y, v_local.z]
            out[i, :3] = [v_local.x, v_local.y, -v_local.z]
        else:
            out[i, :3] = [-v_local.x, v_local.y, -v_local.z]

    # 2) 四元数旋转到局部系
    for i, q in enumerate(rel_path[:, 3:7]):
        q_world = mn.Quaternion(mn.Vector3(q[1:]), q[0])
        q_local = R_inv * q_world
        out[i, 3:7] = [q_local.scalar,
                       q_local.vector.x,
                       q_local.vector.y,
                       q_local.vector.z]

    # 3) yaw 差值
    out[:, 7] = wrap_pi(rel_path[:, 7] - follow_yaw)
    if type ==1:
        # human_local =  [-human_local.x, human_local.y, human_local.z]
        human_local =  [human_local.x, human_local.y, -human_local.z]
    else:
        human_local =  [-human_local.x, human_local.y, -human_local.z]
    
    return out.astype(np.float32),human_local
def smooth_yaw(actions, window_size=3):
    if len(actions) < window_size:
        return actions
    
    yaw_actions = []
    for action in actions:
        yaw_actions.append((action[:2], action[2]))
    
    smoothed_actions = []
    for i in range(len(yaw_actions)):
        start = max(0, i - window_size // 2)
        end = min(len(yaw_actions), i + window_size // 2 + 1)
        window = yaw_actions[start:end]

        avg_pos = sum([a[0] for a in window]) / len(window)
        avg_yaw = sum([a[1] for a in window]) / len(window)
        smoothed_actions.append((avg_pos, avg_yaw))
    
    final = np.array([np.array([pos[0], pos[1], yaw]) for pos, yaw in smoothed_actions])
    return final

def interpolate_rel_path(rel_path: np.ndarray,
                         chunk_size: int,
                         max_dist: float, for_isaac=False) -> np.ndarray:
    """
    把 (x,z,yaw) 路径插值 / 截断到固定长度 chunk_size.
    rel_path : (...,8) 或 (...,3)
    """
    if rel_path.ndim != 2 or rel_path.shape[1] not in (3, 8):
        raise ValueError("rel_path shape must be (N,3) or (N,8)")

    # 1. 取 x,z,yaw
    data = rel_path[:, [0, 2, 7]] if rel_path.shape[1] == 8 else rel_path.copy()
    if for_isaac:
        data[:, :2] = np.stack([data[:, 1], -data[:, 0]], axis=1)
        data[:, 2] *= -1
    # 2. 特例：全 0 或空
    if data.size == 0 or np.allclose(data, 0):
        return np.zeros((chunk_size, 3), np.float32)

    # 3. 计算沿线累积距离
    diffs  = np.diff(data[:, :2], axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    s_full = np.concatenate(([0], np.cumsum(dists)))        # len = N
    total  = s_full[-1]

    # 若超过 max_dist → 找截断点 (总长≥max_dist)
    if total > max_dist:
        idx = np.searchsorted(s_full, max_dist)
        if idx == len(s_full):
            idx -= 1
        # 截断为 idx+1 个点，并在 idx 点上插入精确 max_dist 位置
        excess = s_full[idx] - max_dist
        if excess > 1e-6 and idx > 0:
            ratio = (dists[idx-1] - excess) / dists[idx-1]
            interp_pt = data[idx-1] + ratio * (data[idx] - data[idx-1])
            data = np.vstack([data[:idx], interp_pt])
            s_full = np.concatenate(([0], np.cumsum(np.linalg.norm(
                np.diff(data[:, :2], axis=0), axis=1))))

        total = max_dist

    # 4. 等间距采样到 chunk_size
    if chunk_size == 1:
        samples = np.array([[0, 0, 0]], np.float32)
    else:
        s_samples = np.linspace(0, total, chunk_size)
        samples = np.zeros((chunk_size, 3), np.float32)
        yaw_src = np.unwrap(data[:, 2])    
        for k, s in enumerate(s_samples):
            idx = np.searchsorted(s_full, s) - 1
            idx = np.clip(idx, 0, len(s_full) - 2)
            seg_len = s_full[idx+1] - s_full[idx]
            if seg_len < 1e-8:
                samples[k] = data[idx]
            else:
                t = (s - s_full[idx]) / seg_len
                samples[k, :2] = data[idx, :2] + t * (data[idx+1, :2] - data[idx, :2])
                # yaw 带环绕，简单线插足够（前提 Δyaw 不跨 ±π）
                # samples[k, 2] = data[idx, 2] + t * (data[idx+1, 2] - data[idx, 2])
                yaw_lin = yaw_src[idx] + t * (yaw_src[idx+1] - yaw_src[idx])   # ② 线性插值
                samples[k, 2] = (yaw_lin + np.pi) % (2 * np.pi) - np.pi   

    return samples.astype(np.float32)

# pixel_point, pixel_goal, idx = farthest_pixel_goal_grounding(
#                     local_traj,
#                     depth=cam_depth,
#                     fx=cam.fx,
#                     fy=cam.fy,
#                     cx=cam.cx,
#                     cy=cam.cy,
#                 )
def farthest_pixel_goal_grounding(
    local_traj,   # (N, 3) 局部轨迹 [x, y, z]
    depth,        # (H, W)
    fx, fy, cx, cy,
    min_depth=0.2,
    max_depth=3.0,
    occlusion_eps=0.05,
):
    H, W = depth.shape

    # ---- 转为相机坐标系 ----
    Z_forward = local_traj[:, 0]   # 前方
    X_right   = -local_traj[:, 1]  # 右方
    Y_down    = -local_traj[:, 2]  # 下方

    x_local = X_right
    y_local = Y_down
    z_local = Z_forward

    # ---- 投影到像素平面 ----
    u = fx * x_local / z_local + cx
    v = fy * y_local / z_local + cy

    u_i = np.round(u).astype(int)
    v_i = np.round(v).astype(int)

    # ---- 筛选有效像素 ----
    valid_mask = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    if not np.any(valid_mask):
        return None, None, None

    u_valid = u_i[valid_mask]
    v_valid = v_i[valid_mask]
    z_valid = z_local[valid_mask]

    # ---- 检查深度是否可见 ----
    depth_val = depth[v_valid, u_valid]
    visibility_mask = (depth_val > min_depth) & (depth_val < max_depth) & (z_valid <= depth_val + occlusion_eps)

    if not np.any(visibility_mask):
        return None, None, None

    u_final = u_valid[visibility_mask]
    v_final = v_valid[visibility_mask]
    z_final = z_valid[visibility_mask]

    # ---- 找到最远点 ----
    farthest_idx_in_visible = np.argmax(z_final)

    # ---- 获取最远点在 original local_traj 中的索引 ----
    valid_indices = np.flatnonzero(valid_mask)
    visible_indices = valid_indices[visibility_mask]
    farthest_idx = visible_indices[farthest_idx_in_visible]

    # ---- 返回最远点局部坐标 和 对应像素 ----
    local_farthest = local_traj[farthest_idx][:2]  # 原始局部轨迹坐标
    u_pixel = int(u_final[farthest_idx_in_visible])
    v_pixel = int(v_final[farthest_idx_in_visible])

    return local_farthest, (u_pixel, v_pixel), farthest_idx


def project_pixel_habitat(cam_depth, habitat_path_np, hpos, fquat_mn, fyaw, height = 1.5):
    reletive_path, huamn_local  = world2local(habitat_path_np,hpos, fquat_mn, fyaw,type=0)
    actions = interpolate_rel_path(reletive_path, 30, 3.0, for_isaac=True)
    actions = smooth_yaw(actions,5)
    
    project_action = actions.copy()
    project_action[:, -1] = 0.0
    if np.all(project_action[:, 0] < 0.2):
        return None, actions, None, huamn_local
    pixel_point, pixel_goal, idx = farthest_pixel_goal_grounding(
                    project_action,
                    depth=cam_depth,
                    fx=600,
                    fy=600,
                    cx=640,
                    cy=360,
                    min_depth=0.0,
                    max_depth=10.0,
                    occlusion_eps=0.05,
                )
    return pixel_goal, actions, idx, huamn_local
    pixels = project_ground_traj_to_pixels(project_action, fx=184.7,
                     fy=184.7,cx=320,cy=240,cam_height=1.5)
    return pixels, actions, None, huamn_local