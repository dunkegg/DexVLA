import os
import h5py
import numpy as np

input_h5 = "/mnt/pfs/3zpd5q/code/zf/raw_data/matterport_data/door.h5"
output_dir = "/mnt/pfs/3zpd5q/code/zf/train_data/process_data"
os.makedirs(output_dir, exist_ok=True)

door_move_h5 = os.path.join(output_dir, "door_move.h5")
door_rotate_h5 = os.path.join(output_dir, "door_rotate.h5")

direction_reasoning = {
    "front": "Move close to a door.",
    "left": "Rotate to locate a door.",
    "right": "Rotate to locate a door."
}

MAX_ACTION_LEN = 30  
MAX_MOVE_DIST = 1.0  

def compute_move_action(real_traj):
    """
    根据全局轨迹生成 move action：
    - yaw 为点到点朝向角
    - 相对局部坐标系（以当前点为原点，初始方向为局部x轴）
    """
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    T = len(real_traj)
    actions_all = np.zeros((T, MAX_ACTION_LEN, 3), dtype=np.float32)

    for t in range(T):
        # ---- Step 1. 取从当前点起的后续轨迹 ----
        traj = real_traj[t:, :3]  # 仅取 x, y, z
        if len(traj) < 2:
            actions_all[t, :, :] = 0.0
            continue

        # ---- Step 2. 累计距离，截断到 1m 内 ----
        diffs_global = np.diff(traj[:, :2], axis=0)
        seg_dists_global = np.linalg.norm(diffs_global, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_dists_global)])
        total_length = cumulative[-1]

        if total_length > MAX_MOVE_DIST:
            valid_idx = np.where(cumulative <= MAX_MOVE_DIST)[0]
            traj = traj[valid_idx] if len(valid_idx) >= 2 else traj[:2]

        # ---- Step 3. 确定局部坐标系（以初始方向为 x 轴） ----
        origin = traj[0]
        dx_init = traj[1, 0] - traj[0, 0]
        dy_init = traj[1, 1] - traj[0, 1]
        init_yaw = np.arctan2(dy_init, dx_init)

        c = np.cos(-init_yaw)
        s = np.sin(-init_yaw)

        dx = traj[:, 0] - origin[0]
        dy = traj[:, 1] - origin[1]
        x_local_all = dx * c - dy * s
        y_local_all = dx * s + dy * c

        # ---- Step 4. 点到点 yaw (局部朝向) ----
        yaws = np.zeros(len(traj))
        if len(traj) > 1:
            diff_xy = np.diff(traj[:, :2], axis=0)
            seg_yaws = np.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            yaws[1:] = [normalize_angle(y - init_yaw) for y in seg_yaws]

        pts_local = np.stack([x_local_all, y_local_all, yaws], axis=1)

        # ---- Step 5. pad / truncate ----
        if len(pts_local) < MAX_ACTION_LEN:
            last_val = pts_local[-1] if len(pts_local) > 0 else [0.0, 0.0, 0.0]
            pts_local = np.vstack([
                pts_local,
                np.tile(last_val, (MAX_ACTION_LEN - len(pts_local), 1))
            ])
        else:
            pts_local = pts_local[:MAX_ACTION_LEN]

        actions_all[t] = pts_local.astype(np.float32)
        actions_all[t, 0, :] = 0.0  # 当前帧归零

    return actions_all



def compute_rotate_action(real_traj):
    """生成 rotate action，x,y=0，只转换 yaw"""
    T = len(real_traj)
    actions_all = np.zeros((T, MAX_ACTION_LEN, 3), dtype=np.float32)
    
    for t in range(T):
        base_yaw = real_traj[t, 3]  
        future_traj = real_traj[t:].copy()
        yaw_list = []
        
        for a in future_traj:
            yaw_new = a[3] - base_yaw 
            yaw_list.append([0.0, 0.0, yaw_new])
        
        if len(yaw_list) < MAX_ACTION_LEN:
            last_val = yaw_list[-1] if len(yaw_list) > 0 else [0.0, 0.0, 0.0]
            yaw_list += [last_val] * (MAX_ACTION_LEN - len(yaw_list))
        else:
            yaw_list = yaw_list[:MAX_ACTION_LEN]
        
        actions_all[t] = np.array(yaw_list, dtype=np.float32)
    
    return actions_all


with h5py.File(input_h5, 'r') as f, \
     h5py.File(door_move_h5, 'w') as move_h5, \
     h5py.File(door_rotate_h5, 'w') as rotate_h5:

    episodes = list(f.keys())
    move_count = 0
    rotate_count = 0

    for ep in episodes:
        instruction = f[ep]['instruction'][()]
        instruction = instruction.decode("utf-8") if isinstance(instruction, bytes) else instruction
        lower_inst = instruction.lower()

        direction_type = None
        for key in direction_reasoning.keys():
            if key in lower_inst:
                direction_type = key
                break

        if direction_type is None or "behind" in lower_inst or direction_type == "back":
            continue

        new_instruction = "Move next to a door in the current environment by rotating or moving."
        substep_reasoning = direction_reasoning[direction_type]

        image_data = f[ep]['image_data'][()]              
        real_traj = f[ep]['real_trajectory_data'][()]     
        ref_traj = f[ep]['reference_trajectory_data'][()] 
        obj_name = f[ep]['obj_name'][()]

        if direction_type == "front":
            target_h5 = move_h5
            ep_id = move_count
            move_count += 1
        else:
            target_h5 = rotate_h5
            ep_id = rotate_count
            rotate_count += 1
            total_len = len(real_traj)
            keep_n = 10 if total_len / 2 > 10 else total_len // 2
            image_data = image_data[:keep_n]
            real_traj = real_traj[:keep_n]
            ref_traj = ref_traj[:keep_n]

        T = len(real_traj)
        if T == 0:
            continue

        # 根据类型计算 action
        if direction_type == "front":
            actions_all = compute_move_action(real_traj)
        else:
            actions_all = compute_rotate_action(real_traj)

        ep_name = f"episode_{ep_id:03d}"
        grp = target_h5.create_group(ep_name)
        grp.create_dataset("image_data", data=image_data, dtype=image_data.dtype)
        grp.create_dataset("real_traj", data=real_traj, dtype=real_traj.dtype)
        grp.create_dataset("reference_trajs", data=ref_traj, dtype=ref_traj.dtype)
        grp.create_dataset("obj_name", data=obj_name, dtype=h5py.string_dtype())
        grp.create_dataset("action", data=actions_all, dtype='float32')
        grp.create_dataset("language_raw", data=new_instruction, dtype=h5py.string_dtype())
        grp.create_dataset("substep_reasonings", data=substep_reasoning, dtype=h5py.string_dtype())

        print(f"Saved {ep_name} ({direction_type}) with T={T}, action shape={actions_all.shape}")

print("Done! door_move.h5 and door_rotate.h5 created with updated action.")