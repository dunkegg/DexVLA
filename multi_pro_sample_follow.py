import sys
import os
import time
import json
import re
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
from tqdm import tqdm

import multiprocessing as mp

# ===============================
# 工具函数
# ===============================

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

def check_episode_validity(obs_ds, threshold: float = 0.3):
    rgb = obs_ds
    height, width = rgb.shape[:2]
    rgb3 = rgb[..., :3]
    num_black_pixels = np.sum(np.all(rgb3 == 0, axis=-1))
    return num_black_pixels < threshold * width * height

# ===============================
# 模块顶层函数
# ===============================
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
                # if episode_count > 10:
                #     return True
                # if scene in entry.get("key"):
                #     episode_count +=1
            except json.JSONDecodeError:
                continue
    return False
def run_scene(worker_id, scene, items, yaml_file_path, num_gpus):
    """
    每个 worker 调用
    """
    # 延迟 import habitat 相关内容，避免 spawn 时问题
    import magnum as mn
    from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene_for_obj
    from human_follower.walk_behavior import walk_along_path
    from human_follower.save_data import save_move_obj_data_to_h5
    from habitat_for_sim.utils.goat import read_yaml, extract_dict_from_folder, get_current_scene, process_episodes_and_goals, convert_to_scene_objects
    from habitat_for_sim.agent.path_generator import generate_path
    from scipy.spatial.transform import Rotation as R
    # 将上级目录加入 Python 搜索路径
    from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene
    from human_follower.walk_behavior import walk_along_path_multi, get_path_with_time, generate_interfere_sample_from_target_path
    from human_follower.human_agent import AgentHumanoid, get_humanoid_id
    from human_follower.save_data import save_output_to_h5, to_quat

    from evaluate_dexvln.record import create_log_json, append_log
    from process_data.visualize_sample_data import render_sequence_and_make_video 

    # 读取 yaml 配置
    cfg = read_yaml(yaml_file_path)

    # ==============================
    # 🔥 EGL-safe GPU 绑定
    # ==============================
    gpu_id = worker_id % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HABITAT_SIM_GPU_DEVICE_ID"] = "0"
    os.environ["MAGNUM_LOG"] = "quiet"

    print(f"[Worker {worker_id}] Scene={scene}, items={len(items)}")
    json_data = cfg.json_file_path
    img_output_dir = cfg.img_output_dir


    agilex_bot = None
    ######################################
    

    

    
    max_episodes = cfg.max_episodes

    all_index = 0

    episodes_count = 0

    with open("character_descriptions_new.json", "r") as f:
        id_dict = json.load(f)
    # name_folders = [f"female_{i}" for i in range(35)] + [f"male_{i}" for i in range(65)]
    name_folders = list(id_dict.keys())
    


    structured_data,  filtered_episodes = process_episodes_and_goals(items)
    
            
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
 
    episodes = convert_to_scene_objects(structured_data, filtered_episodes, pathfinder, min_distance=10, sample_all=True)

    print("begin")
    for episode_idx, episode_data in enumerate(tqdm(episodes, desc=f"Scene {scene}", position=worker_id)):
        episode_id = episode_data["episode_id"]
        # if episodes_count > max_episodes:
        #     break

        if is_in_blacklist(current_scene, episode_id , "scene_episode_blacklist.jsonl"):
            print(f"{current_scene} :  {episode_id} in blacklist")
            continue

        #
        human_fps = 10
        human_speed = 0.7
        followed_path = generate_path_from_scene(episode_data, pathfinder, 10, human_fps, human_speed)
        if followed_path is None:
            add_to_blacklist(current_scene, episode_id , "scene_episode_blacklist.jsonl")
            continue

        # reset humanoid
        try:
            simulator.close()
        except:
            pass
        simulator = load_simulator(cfg,3)

        # ###label

        humanoid_name = random.choice(name_folders)
        # humanoid_name =random.choice(["female_6", "male_5"])
        print(f"Selected humanoid: {humanoid_name}")
        follow_description = id_dict[humanoid_name]["description"]
        if isinstance(follow_description, list):
            follow_description = random.choice(follow_description)
        # 原主目标人
        
        target_humanoid = AgentHumanoid(simulator,base_pos=followed_path[0][0], base_yaw = followed_path[0][2], human_data_root = cfg.human_data ,name = humanoid_name,description = follow_description, is_target=True)
        
        interferer_num = random.randint(0,3)
        all_interfering_humanoids = []
        if cfg.multi_humanoids:
            interferer_list = []
            for idx in range(interferer_num):
                # break
                # max_humanoids[idx].reset(name = get_humanoid_id(humanoid_name))
                interferer_name = get_humanoid_id(name_folders, name_exception = humanoid_name)
                interferer_list.append(interferer_name)
            # for interferer_name in ["female_2", "female_3"]:
            # if humanoid_name == "female_6":
            #     interferer_list =  ["male_5"]
            # else:
            #     interferer_list = ["female_6"]

            for interferer_name in interferer_list:
                interferer_description = id_dict[interferer_name]["description"]
                if isinstance(interferer_description, list):
                    interferer_description = random.choice(interferer_description)
                interferer = AgentHumanoid(simulator, base_pos=mn.Vector3(-5, 0.083, -5), base_yaw = 0, human_data_root = cfg.human_data, name = interferer_name, description = interferer_description, is_target=False)
                all_interfering_humanoids.append(interferer)
    
        reset_state = simulator.agents[0].get_state()

        simulator.agents[0].set_state(reset_state)
        obs = simulator.get_sensor_observations(0)['color_0_0']
        black_threshold = 0.3
        # if cfg.multi_humanoids:
        #     black_threshold = 0.1
        # if not check_episode_validity(obs, threshold=black_threshold):
        #     print("invalid black observations")
        #     os.makedirs("black_obs", exist_ok=True)
        #     imageio.imwrite(f'black_obs/{episode_id}.png', obs)
        #     add_to_blacklist(current_scene, episode_id , "scene_episode_blacklist.jsonl")
        #     continue


        
        print(f"Start ------------------------------ {all_index}")
        interfering_humanoids = None
        if cfg.multi_humanoids:
            # k = random.randint(1, 3) 
            # interfering_humanoids = random.sample(all_interfering_humanoids, k)
            interfering_humanoids = all_interfering_humanoids
            for interfering_humanoid in interfering_humanoids:
                radius = random.uniform(1, 5)
                sample_path = generate_interfere_sample_from_target_path(followed_path,pathfinder, radius=radius)
                list_pos = [[point.x,point.y,point.z] for point in sample_path]
                interfering_path = generate_path(list_pos, pathfinder, num_points_between = 2, resolution=256, visualize=False)
                interfering_path = get_path_with_time(interfering_path, time_step=1/human_fps, speed=0.9)
                interfering_humanoid.reset_path(interfering_path)
            

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

        except Exception as e:
            print(f"ERROR:   {e}")
            continue
        

        # render_sequence_and_make_video(output_data, f"{img_output_dir}/episode_{all_index}",f"{video_output_dir}/episode_{all_index}")

        save_output_to_h5(output_data, f"data/raw_data/single_follow_pixel/{worker_id}_episode_{all_index}.hdf5")
        print(f"Worker {worker_id} Already has {episodes_count} cases")
        episodes_count+=len(output_data["follow_paths"])
        all_index+=1

            

# ===============================
# 多进程入口函数
# ===============================

def worker_entry(args):
    worker_id, scene, items, yaml_file_path, num_gpus = args
    import os
    pid = os.getpid()

    print(f"[PID={pid}] worker_id={worker_id}, scene={scene}")
    run_scene(worker_id, scene, items, yaml_file_path, num_gpus)

# ===============================
# 主函数
# ===============================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True)
    args = parser.parse_args()
    yaml_file_path = args.yaml_file_path

    from habitat_for_sim.utils.goat import read_yaml, extract_dict_from_folder
    cfg = read_yaml(yaml_file_path)
    json_data = cfg.json_file_path
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
    scenes = list(data.items())

    num_gpus = getattr(cfg, "num_gpus", 4)
    num_workers = min(getattr(cfg, "num_workers", 4), len(scenes))

    print(f"[Main] scenes={len(scenes)}, workers={num_workers}, gpus={num_gpus}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        pool.map(
            worker_entry,
            [
                (worker_id, scene, items, yaml_file_path, num_gpus)
                for worker_id, (scene, items) in enumerate(scenes)
            ]
        )

if __name__ == "__main__":
    main()
