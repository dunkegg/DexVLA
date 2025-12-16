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
from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene, generate_full_path_from_coords_with_index

from human_follower.walk_behavior import walk_along_path_multi,walk_along_path, generate_interfere_path_from_target_path, get_path_with_time,generate_interfer_path, generate_interfere_sample_from_target_path
from human_follower.human_agent import AgentHumanoid, get_humanoid_id
from human_follower.save_data import save_output_to_h5, save_walk_data_to_h5, to_quat
from habitat_for_sim.utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from evaluate_dexvln.robot import FakeRobotEnv, qwen2_vla_policy
from evaluate_dexvln.record import create_log_json, append_log
from pathlib import Path

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

def load_viewpoint_pose_map(json_path):
    """
    读取 RXR/matterport 提供的 viewpoint pose 文件
    返回: dict[image_id] = 4x4 位姿矩阵
    """
    import json
    with open(json_path, "r") as f:
        data = json.load(f)

    pose_map = {}
    for item in data:
        image_id = item["image_id"]
        pose = np.array(item["pose"], dtype=np.float32).reshape(4, 4)
        # pose = pose[:3, 3]
        # pose[2,3] = item["height"]
        pose_map[image_id] = [pose,item["height"]]
    return pose_map
def rotmat_to_quat(R):
    """将旋转矩阵转换为四元数 (w,x,y,z)"""
    import transforms3d.quaternions as tq
    return tq.mat2quat(R)  # (w,x,y,z)

def pano_id_to_xyz(pose_map, pano_id):
    """
    pano_id 就是 image_id 字符串
    返回: world_pos, world_quat
    """
    M = pose_map[pano_id][0]     # 4x4 世界坐标系位姿矩阵
    height = pose_map[pano_id][1] -2
    # 世界坐标
    # tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
    tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
    tz = height
    pos = np.array([tx, tz, -ty], dtype=np.float32)

    
    # 旋转矩阵 → 四元数
    R = M[:3, :3]
    quat = rotmat_to_quat(R)
    
    return pos, quat
def convert_path_to_coords(pose_map, pano_list):
    coords = []

    for pid in pano_list:
        pos, quat = pano_id_to_xyz(pose_map, pid)
        if pos is not None:
            coords.append(pos)
        else:
            print(f"Warning: pano id {pid} not found in scene!")
    return coords

def convert_all_segments(instruction_blocks, index_map):
    """
    instruction_blocks: list of lists
        [
            [
                { 'segment': [a,b], 'text': ... },
                { 'segment': [c,d], 'text': ... },
            ],
            [
                { 'segment': [e,f], 'text': ... },
                ...
            ]
        ]
    
    index_map: coords 分段对应 full_path 的索引
        index_map[i] = (full_start, full_end)

    返回结构化的新 instructions：
        segment 替换成 full_path 的全局 index
    """

    def convert_segment(seg, index_map):
        a, b = seg
        full_start = index_map[a][0]
        full_end = index_map[b - 1][1]
        return [full_start, full_end]

    new_blocks = []

    for block in instruction_blocks:
        new_block = []
        for entry in block:
            old_seg = entry["segment"]
            new_seg = convert_segment(old_seg, index_map)

            new_block.append({
                "segment": new_seg,
                "text": entry["text"]
            })

        new_blocks.append(new_block)

    return new_blocks

def parse_instruction_text(instr: str):
    """
    解析一条指令文本，如：
    "0-1: Go forward ...\n1-4: Turn right ..."

    返回一个列表，每个元素包含：
    {
        "segment": [start_idx, end_idx],
        "text": "动作描述"
    }
    """
    results = []
    
    # 按换行拆分
    lines = instr.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 使用正则匹配 "num-num: description"
        match = re.match(r"(\d+)\s*-\s*(\d+)\s*:\s*(.*)", line)
        if match:
            start_idx = int(match.group(1))
            end_idx = int(match.group(2))
            text = match.group(3).strip()

            results.append({
                "segment": [start_idx, end_idx],
                "text": text
            })
        else:
            # 如果没有编号（极少数情况），也加入
            results.append({
                "segment": None,
                "text": line
            })

    return results


def parse_instructions_block(instructions_list):
    """
    instructions_list 是一个列表，包含多条说明，例如：
    [
        "0-1: Go forward ...",
        "0-1: Walk towards ...",
        ...
    ]

    返回一个列表，每条说明被解析好的结构化信息
    """
    parsed = []
    for instr in instructions_list:
        parsed.append(parse_instruction_text(instr))
    return parsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True,
                        help='Path to the YAML config file')
    args = parser.parse_args()

    cfg = read_yaml(args.yaml_file_path)
    img_output_dir = cfg.img_output_dir
    video_output_dir = cfg.video_output_dir
    log_path = create_log_json() if cfg.log_path is None else cfg.log_path 

    agilex_bot = None
    ######################################
    

    

    # cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)

    with open("habitat_for_sim/RxR_10000.json", "r") as f:
        scan_dict = json.load(f)
        
    max_episodes = cfg.max_episodes

    all_index = 0
    success_count = 0
    episodes_count = 0
    jump_idx = get_max_episode_number(img_output_dir)+1
    jump_idx = 0

    for scan_id, entries in scan_dict.items():
        # if episodes_count > max_episodes:
        #     break


        cfg.current_scene = scan_id
        
        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass

        simulator = load_simulator(cfg)
        _ , pose_data_path =  find_scene_path(cfg, scan_id)
        pose_map = load_viewpoint_pose_map(pose_data_path)
        semantic_scene = simulator.semantic_scene
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        if not simulator.pathfinder.is_loaded:
            print("Failed to load or generate navmesh.")
            continue
            raise RuntimeError("Failed to load or generate navmesh.")   

        if scan_id == '29hnd4uzFmX':
            continue 
        print("begin")
        for entry in entries:

            # if episodes_count > max_episodes:
            #     break

            if all_index < jump_idx:
                all_index += 1
                continue  

            # # if followed_path is None:
            # #     add_to_blacklist(current_scene, episode_id , "scene_episode_blacklist.jsonl")
            # #     continue

            # # reset humanoid
            # try:
            #     simulator.close()
            # except:
            #     pass
            # simulator = load_simulator(cfg)
            pano_ids = entry["path"]
            
            coords = convert_path_to_coords(pose_map, pano_ids)
            obs_fps = 10
            robot_speed = 0.7
            followed_path, index_map = generate_full_path_from_coords_with_index(coords, pathfinder,obs_fps, robot_speed)
            if followed_path is None:
                continue
            instructions = entry["instructions"]
            instruction_blocks = parse_instructions_block(instructions)
            new_instructions_map=convert_all_segments(instruction_blocks, index_map)
            # ###label

            print(f"Start ------------------------------ {all_index}")


            # try:
            output_data = walk_along_path(
                all_index=all_index,
                sim=simulator,
                walk_path=followed_path,
                fps=10,
                forward_speed=robot_speed,
                timestep_gap = 1/obs_fps, 
                robot = agilex_bot
            )

            # except Exception as e:
            #     print(f"ERROR:   {e}")
            #     continue

            
            save_walk_data_to_h5(output_data["obs"], walk_path= followed_path, instruction_map = new_instructions_map, h5_path=f"data/raw_data/rxr/episode_{all_index}.hdf5")
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
            print(f"Already has {episodes_count} cases")
            episodes_count+=1
            all_index+=1

            print(f"Success Rate: {success_count/all_index}")
            


