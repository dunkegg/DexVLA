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
from habitat_for_sim.utils.goat import read_yaml, extract_dict_from_folder, get_current_scene, process_episodes_and_goals, convert_to_scene_objects, find_scene_path, calculate_euclidean_distance
from habitat_for_sim.agent.path_generator import generate_path
from habitat_for_sim.utils.frontier_exploration import FrontierExploration
from scipy.spatial.transform import Rotation as R
# 将上级目录加入 Python 搜索路径
from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene

from human_follower.walk_behavior import walk_along_path_multi, generate_interfere_path_from_target_path, get_path_with_time,generate_interfer_path, generate_interfere_sample_from_target_path
from human_follower.human_agent import AgentHumanoid, get_humanoid_id

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

    # 如果文件存在就读取，否则新建
    if os.path.exists(blacklist_path):
        with open(blacklist_path, 'r') as f:
            blacklist = json.load(f)
    else:
        blacklist = {}

    blacklist[key] = True

    with open(blacklist_path, 'w') as f:
        json.dump(blacklist, f, indent=2)
    print(f"✅ 已加入黑名单: {key}")

def is_in_blacklist(scene: str, idx: int, blacklist_path: str) -> bool:
    key = make_key(scene, idx)

    if not os.path.exists(blacklist_path):
        return False

    with open(blacklist_path, 'r') as f:
        blacklist = json.load(f)

    return key in blacklist

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
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 16
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        "model_path": cfg.model_path,
        #############################################################################
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
    }

    # fake env for debug
    policy = qwen2_vla_policy(policy_config)
    agilex_bot = FakeRobotEnv(policy_config, policy,plot_dir=img_output_dir)
    ######################################
    

    
    # 初始化目标文件列表
    target_files = []   

    # 遍历文件夹并将相对路径添加到目标文件列表
    for root, dirs, files in os.walk(json_data):
        for file in files:
            # 计算相对路径并加入列表
            relative_path = os.path.relpath(os.path.join(root, file), json_data)
            target_files.append(relative_path)
    
    
    # cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)

    data = extract_dict_from_folder(json_data, target_files)
    
    max_episodes = cfg.max_episodes

    all_index = 0
    success_count = 0
    episodes_count = 0
    jump_idx = get_max_episode_number(img_output_dir)+1
    for file_name, content in sorted(data.items()):
        if episodes_count > max_episodes:
            break
        if all_index < jump_idx:
            all_index += 1
            continue  
        structured_data,  filtered_episodes = process_episodes_and_goals(content)
        episodes = convert_to_scene_objects(structured_data, filtered_episodes, ogn=False)
                
        cfg.current_scene = current_scene = get_current_scene(structured_data)
        
        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass

        simulator = load_simulator(cfg)
        
        semantic_scene = simulator.semantic_scene
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        if not simulator.pathfinder.is_loaded:
            print("Failed to load or generate navmesh.")
            continue
            raise RuntimeError("Failed to load or generate navmesh.")   


        with open("character_descriptions.json", "r") as f:
            id_dict = json.load(f)

        # ###label
        # folders = [f"female_{i}" for i in range(35)] + [f"male_{i}" for i in range(65)]
        # humanoid_name = folders[all_index]

        
        humanoid_name = get_humanoid_id(id_dict, name_exception=None) 
        follow_description = id_dict[humanoid_name]["description"]
        if not cfg.multi_humanoids:
            humanoid_name = "female_0"
            follow_description = None
        # 原主目标人
        
        target_humanoid = AgentHumanoid(simulator,base_pos=mn.Vector3(-5, 0.083, -5), base_yaw = 0, human_data_root = cfg.human_data ,name = humanoid_name,description = follow_description, is_target=True)
        
        all_interfering_humanoids = []
        if cfg.multi_humanoids:
            for idx in range(3):
                # break
                # max_humanoids[idx].reset(name = get_humanoid_id(humanoid_name))
                interferer_name = get_humanoid_id(id_dict, name_exception = humanoid_name)
                interferer_description = id_dict[humanoid_name]["description"]
                interferer = AgentHumanoid(simulator, base_pos=mn.Vector3(-5, 0.083, -5), base_yaw = 0, human_data_root = cfg.human_data, name = interferer_name, description = interferer_description, is_target=False)
                all_interfering_humanoids.append(interferer)

    
        reset_state = simulator.agents[0].get_state()
        

        print("begin")
        for episode_idx, episode_data in enumerate(tqdm(episodes)):
            episode_id = episode_data["episode_id"]
            if episodes_count > max_episodes:
                break

            if is_in_blacklist(current_scene, episode_id , "scene_episode_blacklist.json"):
                print(f"{current_scene} :  {episode_id} in blacklist")
                continue

            if all_index < jump_idx:
                all_index += 1
                continue  

            simulator.agents[0].set_state(reset_state)
            obs = simulator.get_sensor_observations(0)['color_0_0']
            black_threshold = 0.3
            # if cfg.multi_humanoids:
            #     black_threshold = 0.1
            if not check_episode_validity(obs, threshold=black_threshold):
                print("invalid black observations")
                os.makedirs("black_obs", exist_ok=True)
                imageio.imwrite(f'black_obs/{episode_id}.png', obs["color_0_0"])
                add_to_blacklist(current_scene, episode_id , "scene_episode_blacklist.json")
                continue
            #

            human_fps = 10
            human_speed = 0.7
            followed_path = generate_path_from_scene(episode_data, pathfinder, 10, human_fps, human_speed)
            if followed_path is None:
                add_to_blacklist(current_scene, episode_id , "scene_episode_blacklist.json")
                continue
            
            print(f"Start ------------------------------ {all_index}")
            interfering_humanoids = None
            if cfg.multi_humanoids:
                k = random.randint(1, 3) 
                interfering_humanoids = random.sample(all_interfering_humanoids, k)
                ##
                for interfering_humanoid in interfering_humanoids:
                    sample_path = generate_interfere_sample_from_target_path(followed_path,pathfinder, 1)
                    list_pos = [[point.x,point.y,point.z] for point in sample_path]
                    interfering_path = generate_path(list_pos, pathfinder, visualize=False)
                    interfering_path = get_path_with_time(interfering_path, time_step=1/human_fps, speed=0.9)
                    interfering_humanoid.reset_path(interfering_path)
                

            agilex_bot.reset(simulator.agents[0],n_frames=8, human_description=follow_description)
            try:
                output_data = walk_along_path_multi(
                    all_index=all_index,
                    sim=simulator,
                    humanoid_agent=target_humanoid,
                    human_path=followed_path,
                    fps=10,
                    forward_speed=human_speed,
                    timestep_gap = 1/human_fps, 
                    interfering_humanoids=interfering_humanoids,
                    robot = agilex_bot
                )
                append_log(log_path, index=all_index, success=output_data["follow_result"], sample_fps=output_data["sample_fps"], plan_fps=output_data["plan_fps"], follow_size=output_data["follow_size"])
                if output_data["follow_result"]:
                    success_count+=1
            except Exception as e:
                print(e)
                continue

            
            video_output = video_output_dir
            os.makedirs(video_output, exist_ok=True)
            vut.make_video(
                output_data["obs"],
                "color_0_0",
                "color",
                f"{video_output}/humanoid_wrapper_{all_index}",
                open_vid=False,
            )
            print(f"Case {all_index}, {humanoid_name} Done, Already has {episodes_count} cases")
            all_index+=1

            print(f"Success Rate: {success_count/all_index}")
            


