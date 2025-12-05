import sys
import os
import quaternion as qt
import random
import numpy as np
import h5py
import cv2
import json
import magnum as mn
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils import viz_utils as vut
print(habitat_sim.__file__)
import argparse
import imageio
from habitat_for_sim.utils.goat import read_yaml, extract_dict_from_folder, get_current_scene, process_episodes_and_goals, convert_to_scene_objnav_rotate, find_scene_path, calculate_euclidean_distance
from habitat_for_sim.agent.path_generator import generate_path
from habitat_for_sim.utils.frontier_exploration import FrontierExploration
from scipy.spatial.transform import Rotation as R
# 将上级目录加入 Python 搜索路径
from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene_for_obj

from human_follower.walk_behavior import walk_along_path_multi, walk_along_path,rotate_to_target, generate_interfere_path_from_target_path, get_path_with_time,generate_interfer_path, generate_interfere_sample_from_target_path
from human_follower.human_agent import AgentHumanoid, get_humanoid_id
from human_follower.save_data import save_output_to_h5, save_walk_data_to_h5, to_quat, save_rotate_obj_data_to_h5, save_move_obj_data_to_h5
from habitat_for_sim.utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from evaluate_dexvln.robot import FakeRobotEnv, qwen2_vla_policy
from evaluate_dexvln.record import create_log_json, append_log

import json
from collections import defaultdict

def load_json_group_by_scene(json_path):
    from collections import defaultdict
    import json

    scene_dict = defaultdict(list)

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            scene = item.get("current_scene")
            scene_dict[scene].append(item)

    return dict(scene_dict)



def time_ms():
    return time.time_ns() // 1_000_000

import re

def get_max_episode_number(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    max_num = -1
    for name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, name)):
            match = re.match(r'episode_(\d+)', name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    return max_num

def check_episode_validity(obs_ds, threshold: float = 0.3):
    """检查前 max_check_frames 帧是否有效（大面积黑图则无效）"""

    rgb = obs_ds
    height, width = rgb.shape[:2]  # 自动读取图像高宽
    rgb3 = rgb[..., :3]  # 只取前三通道
    num_black_pixels = np.sum(np.all(rgb3 == 0, axis=-1))
    # num_black_pixels = np.sum(np.sum(rgb, axis=-1) == 0)
    if num_black_pixels >= threshold * width * height:
        return False  # 当前帧是大面积黑图
    
    return True
import math
def distance(p1, p2):
    """计算二维点距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def add_path_if_valid(path_dict, start_pos, goal_pos, threshold=1.0):
    """
    start_pos: (sx, sy)
    goal_pos: (gx, gy)
    threshold: 距离阈值（米）
    
    返回:
        True  -> 有效并已加入字典
        False -> 与已有路径太近，无效
    """

    for pid, data in path_dict.items():
        old_start = data["start"]
        old_goal = data["goal"]

        # 起点和终点同时在1m内 → 无效
        if distance(start_pos, old_start) < threshold and \
           distance(goal_pos, old_goal) < threshold:
            return False

    # 有效 → 加入字典（ID 可按长度自动编号）
    new_id = len(path_dict)
    path_dict[new_id] = {
        "start": start_pos,
        "goal": goal_pos
    }
    return True

def quaternion_to_list(q):
    import numpy as np
    import quaternion
    # 如果是 numpy.quaternion 类型
    if isinstance(q, quaternion.quaternion):
        return list(q.components)  # [w, x, y, z]
    # 如果是 list 或 np.ndarray 类型，直接返回
    elif isinstance(q, (list, np.ndarray)):
        return list(q)
    else:
        raise TypeError(f"Unsupported rotation type: {type(q)}")
     
if __name__ == '__main__':
    valid_json_path = "valid_candidates.jsonl"
    save_json = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True,
                        help='Path to the YAML config file')
    args = parser.parse_args()

    cfg = read_yaml(args.yaml_file_path)
    move_data_path = cfg.move_data_path
    data = load_json_group_by_scene(move_data_path)
    img_output_dir = cfg.img_output_dir
    video_output_dir = cfg.video_output_dir
    log_path = create_log_json() if cfg.log_path is None else cfg.log_path 

    agilex_bot = None

    # 初始化目标文件列表
    
    max_episodes = cfg.max_episodes
    all_index = 0
    jump_idx = get_max_episode_number(img_output_dir)+1
    jump_idx = 0

    for scene, items in sorted(data.items()):
        cfg.current_scene = current_scene = scene
        try:
            simulator.close()
        except:
            pass    
        simulator = load_simulator(cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)

        for item in items:
            start_pose = item["start"]
            goal_pose = item["goal"] 

            reset_state = simulator.agents[0].get_state()
            reset_state.position = np.array(start_pose["position"], dtype=np.float32)
            # quat_np_start = to_quat(start_pose["rotation"])
            # quat_qt_start = qt.quaternion(quat_np_start[0], quat_np_start[1], quat_np_start[2], quat_np_start[3])
            reset_state.rotation = goal_pose["rotation"]
            simulator.agents[0].set_state(reset_state)
            obs = simulator.get_sensor_observations(0)['color_0_0']

            os.makedirs("/mnt/pfs/s7fsio/code/eval/DexVLA/results_eval_obj/image", exist_ok=True)
            img_path = os.path.join("/mnt/pfs/s7fsio/code/eval/DexVLA/results_eval_obj/image", f"start.png")
            imageio.imwrite(img_path, obs)
            
            # reset_state = simulator.agents[0].get_state()
            # reset_state.position = np.array(goal_pose["position"], dtype=np.float32)
            # reset_state.rotation = goal_pose["rotation"]
            # simulator.agents[0].set_state(reset_state)
            # obs = simulator.get_sensor_observations(0)['color_0_0']
            # os.makedirs("/mnt/pfs/s7fsio/code/eval/DexVLA/results_eval_obj/image", exist_ok=True)
            # img_path = os.path.join("/mnt/pfs/s7fsio/code/eval/DexVLA/results_eval_obj/image", f"end.png")
            # imageio.imwrite(img_path, obs)

            black_threshold = 0.3
            if not check_episode_validity(obs, threshold=black_threshold):
                # print("invalid black observations")
                # os.makedirs("black_obs", exist_ok=True)
                # imageio.imwrite(f'black_obs/{episode_id}.png', obs)
                # add_to_blacklist(current_scene, episode_id , "scene_episode_blacklist_obj.jsonl")
                continue
            
            obs_fps = 10
            robot_speed = 0.7
            try:
                followed_path = generate_path_from_scene_for_obj(item, pathfinder, 5, obs_fps, robot_speed)
            except Exception as e:
                print(f"ERROR:   {e}")
                continue


            try:
                output_data = walk_along_path(
                    all_index=all_index,
                    sim=simulator,
                    walk_path=followed_path,
                    fps=10,
                    forward_speed=robot_speed,
                    timestep_gap = 1/obs_fps, 
                    robot = agilex_bot
                )
            except Exception as e:
                print(f"ERROR:   {e}")
                continue

            # if len(output_data["obs"]) == 0: continue
            # 保存数据
            save_move_obj_data_to_h5(output_data["obs"], 
                                 walk_path= followed_path, 
                                 h5_path=f"data/raw_data/obj/move/episode_{all_index}.hdf5",
                                 item = item)
            if all_index < 50:
                video_output = video_output_dir
                os.makedirs(video_output, exist_ok=True)
                vut.make_video(
                    output_data["obs"],
                    "color_0_0",
                    "color",
                    f"{video_output}/humanoid_wrapper_{all_index}",
                    open_vid=False,
            )

            all_index += 1
        print(f"Case {all_index}")
    print(f"Case {all_index}")

            


