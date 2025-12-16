import os
import json
import yaml
import random
import imageio
import argparse
import numpy as np
import magnum as mn
import matplotlib.pyplot as plt
from collections import defaultdict
from habitat_sim.utils import viz_utils as vut
from human_follower.walk_behavior import walk_along_path
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from habitat_for_sim.utils.load_scene import load_simulator

class Config:
    def __init__(self, config_dict):
        self._config = config_dict

    def __getattr__(self, name):
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __setattr__(self, name, value):
        if name == "_config":
            super().__setattr__(name, value)
        else:
            self._config[name] = value

    def to_dict(self):
        return self._config
    
def read_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return Config(data)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None

def habitat_quat_to_magnum(q_np):
    # q_np: numpy.quaternion (w, x, y, z)
    return mn.Quaternion(
        mn.Vector3(q_np.x, q_np.y, q_np.z),
        q_np.w
    )

# def local2global(reset_state, local_path):
#     out = local_path.copy()
#     N = len(out)

#     curr_pos = np.array(reset_state.position, dtype=np.float32)
#     curr_rot = reset_state.rotation
#     q = habitat_quat_to_magnum(curr_rot)

#     # 存储每个点对应的四元数 (list [x, y, z, w])
#     quat_list = []

#     # 坐标转换
#     for i, row in enumerate(local_path):
#         v_local = mn.Vector3(row[0], row[1], row[2])
#         v_world = q.transform_vector(v_local) + mn.Vector3(*curr_pos)
#         out[i, 0] = v_world.x
#         out[i, 1] = curr_pos[1]  
#         out[i, 2] = v_world.z

#     for i in range(N - 1):
#         dx = out[i+1, 0] - out[i, 0]
#         dz = out[i+1, 2] - out[i, 2]
#         yaw = np.arctan2(dz, dx)
#         out[i, 3] = yaw

#         q_yaw = quat_from_angle_axis(yaw, np.array([0, 1, 0]))
#         quat_list.append([q_yaw.x, q_yaw.y, q_yaw.z, q_yaw.w])

#     if N > 1:
#         out[-1, 3] = out[-2, 3]
#         quat_list.append(quat_list[-1])
#     else:
#         # 如果只有一个点
#         q_yaw = quat_from_angle_axis(0.0, np.array([0, 1, 0]))
#         quat_list.append([q_yaw.x, q_yaw.y, q_yaw.z, q_yaw.w])

#     return out.astype(np.float32), quat_list



def local2global(reset_state, local_path, rotation_step=np.pi/8):
    import numpy as np
    import magnum as mn
    from habitat_sim.utils.common import quat_from_angle_axis

    out = local_path.copy()
    N = len(out)

    curr_pos = np.array(reset_state.position, dtype=np.float32)
    curr_rot = reset_state.rotation
    q = mn.Quaternion((curr_rot.x, curr_rot.y, curr_rot.z), curr_rot.w)

    quat_list = []

    # 坐标转换
    for i, row in enumerate(local_path):
        v_local = mn.Vector3(row[0], row[1], row[2])
        v_world = q.transform_vector(v_local) + mn.Vector3(*curr_pos)
        out[i, 0] = v_world.x
        out[i, 1] = curr_pos[1]
        out[i, 2] = v_world.z

    # 初始 yaw
    prev_yaw = 0.0
    prev_q = quat_from_angle_axis(prev_yaw, np.array([0, 1, 0]))

    for i in range(N - 1):
        dx = out[i+1, 0] - out[i, 0]
        dz = out[i+1, 2] - out[i, 2]
        goal_yaw = np.arctan2(dz, dx)
        # goal_yaw = np.arctan2(-dx, dz)

        # --- 限制每步 yaw 旋转，保证方向与前一四元数一致 ---
        angle_diff = (goal_yaw - prev_yaw + np.pi) % (2*np.pi) - np.pi
        delta = np.clip(angle_diff, -rotation_step, rotation_step)
        current_yaw = prev_yaw + delta

        # 生成四元数
        q_yaw = quat_from_angle_axis(current_yaw, np.array([0, 1, 0]))

        # 保存
        out[i, 3] = current_yaw
        quat_list.append([q_yaw.x, q_yaw.y, q_yaw.z, q_yaw.w])

        prev_yaw = current_yaw
        prev_q = q_yaw

    # 最后一个点
    if N > 1:
        out[-1, 3] = out[-2, 3]
        quat_list.append(quat_list[-1])
    else:
        q_yaw = quat_from_angle_axis(0.0, np.array([0, 1, 0]))
        quat_list.append([q_yaw.x, q_yaw.y, q_yaw.z, q_yaw.w])

    return out.astype(np.float32), quat_list



def generate_local_trajectory(points_xz, resolution=0.1):
    points = np.array(points_xz, dtype=np.float32)
    traj = []

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]

        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)

        if seg_len < 1e-6:
            continue 
        num_samples = int(np.floor(seg_len / resolution))

        direction = seg_vec / seg_len

        for k in range(num_samples):
            pos = p0 + direction * (k * resolution)
            traj.append([pos[0], 0.0, pos[1], 0.0])

    last = points[-1]
    traj.append([last[0], 0.0, last[1], 0.0])

    return np.array(traj, dtype=np.float32)

def load_json_group_by_scene(json_path):
    scene_dict = defaultdict(list)
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            scene = item.get("current_scene")
            scene_dict[scene].append(item)

    return dict(scene_dict)


def visualize_trajectories(local_path, global_path, save_path="trajectory_plot.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.set_title("Local Trajectory (x-z)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")

    ax1.plot(local_path[:,0], local_path[:,2], 'o-', color='blue', label='local path')
    ax1.scatter(local_path[:,0], local_path[:,2], color='lightblue', s=20)
    ax1.scatter(local_path[0,0], local_path[0,2], color='red', s=80, marker='o', label='local start')

    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')

    ax2.set_title("Global Trajectory (x-z)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    ax2.plot(global_path[:,0], global_path[:,2], 'o-', color='green', label='global path')
    ax2.scatter(global_path[:,0], global_path[:,2], color='lightgreen', s=20)
    ax2.scatter(global_path[0,0], global_path[0,2], color='red', s=80, marker='o', label='global start')

    yaws = global_path[:, 3]
    arrow_len = 0.1

    dx = np.cos(yaws) * arrow_len
    dz = np.sin(yaws) * arrow_len

    ax2.quiver(
        global_path[:,0], global_path[:,2],    # starting points
        dx, dz,                                # arrow directions
        angles='xy', scale_units='xy', scale=1, 
        color='black', width=0.005, label='yaw direction'
    )

    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory visualization saved to {save_path}")


if __name__ == '__main__':
    random.seed(666)
    fps=10
    move_data_path = "/mnt/pfs/3zpd5q/code/eval/DexVLA/valid_candidates.jsonl"
    data = load_json_group_by_scene(move_data_path)
    local_path = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],
                  [0.0, 0.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [0.0, 0.0]]

    sample_num = 5
    save_dir = "/mnt/pfs/3zpd5q/code/eval/DexVLA/test_habitat/image"
    os.makedirs(save_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True,
                        help='Path to the YAML config file')
    args = parser.parse_args()
    cfg = read_yaml(args.yaml_file_path)
    scene = random.choice(list(data.keys()))
    cfg.current_scene = scene
    simulator = load_simulator(cfg)

    for idx in range(sample_num):
        sample = random.choice(data[scene])
        # 设置 agent 状态
        reset_state = simulator.agents[0].get_state()
        reset_state.position = np.array(sample["start"]["position"], dtype=np.float32)
        reset_state.rotation = sample["goal"]["rotation"]
        simulator.agents[0].set_state(reset_state)
        loc_path = local_path.copy()
        # 生成局部轨迹和全局轨迹
        loc_path = generate_local_trajectory(loc_path)
        global_path, quats = local2global(reset_state, loc_path)

        observations = []
        for time_step in range(len(global_path)):
            follow_state = simulator.agents[0].get_state()
            follow_state.position = global_path[time_step, :3]
            follow_state.rotation = quats[time_step]
            
            # 更新物理引擎
            simulator.step_physics(1.0 / fps)
            simulator.agents[0].set_state(follow_state)
            obs = simulator.get_sensor_observations(0)
            observations.append(obs)
            
        vut.make_video(
            observations,
            "color_0_0",
            "color",
            f"{save_dir}/humanoid_wrapper_{idx}",
            open_vid=False,
        )
        save_path = os.path.join(save_dir, f"traj_{idx+1}.png")
        visualize_trajectories(loc_path, global_path, save_path=save_path)

            