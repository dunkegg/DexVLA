import sys
import os
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
from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene

from human_follower.walk_behavior import walk_along_path_multi, walk_along_path,rotate_to_target, generate_interfere_path_from_target_path, get_path_with_time,generate_interfer_path, generate_interfere_sample_from_target_path
from human_follower.human_agent import AgentHumanoid, get_humanoid_id
from human_follower.save_data import save_output_to_h5, save_walk_data_to_h5, to_quat, save_rotate_obj_data_to_h5
from habitat_for_sim.utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from evaluate_dexvln.robot import FakeRobotEnv, qwen2_vla_policy
from evaluate_dexvln.record import create_log_json, append_log


def make_key(scene: str, idx: int) -> str:
    return f"{scene}_{idx}"
def add_to_blacklist(scene: str, idx: int, blacklist_path: str):
    key = make_key(scene, idx)

    # 若已存在就不重复写入
    if is_in_blacklist(scene, idx, blacklist_path):
        print(f"⚠️ 已存在于黑名单: {key}")
        return

    with open(blacklist_path, 'a') as f:
        f.write(json.dumps({"key": key}) + '\n')
    print(f"✅ 已加入黑名单: {key}")

def is_in_blacklist(scene: str, idx: int, blacklist_path: str) -> bool:
    key = make_key(scene, idx)
    if not os.path.exists(blacklist_path):
        return False

    with open(blacklist_path, 'r') as f:
        episode_count = 0
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("key") == key:
                    return True
                if episode_count > 10:
                    return True
                if scene in entry.get("key"):
                    episode_count +=1
            except json.JSONDecodeError:
                continue
    return False
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True,
                        help='Path to the YAML config file')
    args = parser.parse_args()

    cfg = read_yaml(args.yaml_file_path)
    json_data = cfg.json_file_path
    img_output_dir = cfg.img_output_dir
    video_output_dir = cfg.video_output_dir
    log_path = create_log_json() if cfg.log_path is None else cfg.log_path 

    agilex_bot = None

    # 初始化目标文件列表
    target_files = []   
    for root, dirs, files in os.walk(json_data):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), json_data)
            target_files.append(relative_path)

    data = extract_dict_from_folder(json_data, target_files)
    
    max_episodes = cfg.max_episodes
    all_index = 0
    success_count = 0
    episodes_count = 0
    jump_idx = get_max_episode_number(img_output_dir)+1
    # jump_idx = 498

    with open("character_descriptions_new.json", "r") as f:
        id_dict = json.load(f)
    name_folders = list(id_dict.keys())

    for file_name, content in sorted(data.items()):
        structured_data, filtered_episodes = process_episodes_and_goals(content)
        cfg.current_scene = current_scene = get_current_scene(structured_data)
        
        # 初始化模拟器
        try:
            simulator.close()
        except:
            pass
        simulator = load_simulator(cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        if not simulator.pathfinder.is_loaded:
            print("Failed to load or generate navmesh.")
            continue

        # 获取 episode 候选点
        episodes = convert_to_scene_objnav_rotate(structured_data, filtered_episodes, pathfinder, 
                                                  min_distance=3.0, max_distance=5.0, sample_all=True)

        print("begin rotation data collection")
        for episode_idx, episode_data in enumerate(tqdm(episodes)):
            episode_id = episode_data["episode_id"]

            if all_index < jump_idx:
                all_index += 1
                continue  
            if is_in_blacklist(current_scene, episode_id , "scene_episode_blacklist_obj.jsonl"):
                # print(f"{current_scene} :  {episode_id} in blacklist")
                continue


            # reset humanoid
            try:
                simulator.close()
            except:
                pass
            simulator = load_simulator(cfg)
            reset_state = simulator.agents[0].get_state()
            reset_state.position = np.array(episode_data.start_position)
            reset_state.rotation = np.array(episode_data.start_rotation)
            simulator.agents[0].set_state(reset_state)

            obs = simulator.get_sensor_observations(0)['color_0_0']
            black_threshold = 0.3
            if not check_episode_validity(obs, threshold=black_threshold):
                print("invalid black observations")
                os.makedirs("black_obs", exist_ok=True)
                imageio.imwrite(f'black_obs/{episode_id}.png', obs)
                add_to_blacklist(current_scene, episode_id , "scene_episode_blacklist_obj.jsonl")
                continue

            output_data = rotate_to_target(
                all_index=all_index,
                simulator=simulator,
                target_pos=np.array(episode_data.goal["position"]),
                fps=10,
                timestep_gap=1/10,
                robot=agilex_bot,
                angle_threshold=0.17,
                rotation_step=np.pi/16
            )

            # 保存数据
            save_rotate_obj_data_to_h5(
                output_data["obs"], output_data["trajectory"], 
                h5_path=f"data/raw_data/obj/episode_{all_index}.hdf5", episode_data = episode_data)
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

            print(f"Case {all_index}, Already has {episodes_count} cases")
            episodes_count += 1
            all_index += 1
            print(f"Success Rate: {success_count/all_index}")

            


