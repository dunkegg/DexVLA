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

def run_scene(worker_id, scene, items, yaml_file_path, num_gpus):
    """
    每个 worker 调用
    """
    # 延迟 import habitat 相关内容，避免 spawn 时问题
    import magnum as mn
    from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene_for_obj
    from human_follower.walk_behavior import walk_along_path
    from human_follower.save_data import save_move_obj_data_to_h5
    from habitat_for_sim.utils.goat import read_yaml

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

    cfg.current_scene = scene
    simulator = load_simulator(cfg)
    pathfinder = simulator.pathfinder
    pathfinder.seed(cfg.seed)

    all_index = 0

    for item in tqdm(items, desc=f"Scene {scene}", position=worker_id):
        try:
            start_pose = item["start"]
            goal_pose = item["goal"]

            reset_state = simulator.agents[0].get_state()
            reset_state.position = np.array(start_pose["position"], dtype=np.float32)
            reset_state.rotation = goal_pose["rotation"]
            simulator.agents[0].set_state(reset_state)

            obs = simulator.get_sensor_observations(0)['color_0_0']

            if not check_episode_validity(obs, threshold=0.3):
                continue

            followed_path = generate_path_from_scene_for_obj(
                item,
                pathfinder,
                5,
                cfg.obs_fps,
                cfg.robot_speed,
            )

            output_data = walk_along_path(
                all_index=all_index,
                sim=simulator,
                walk_path=followed_path,
                fps=cfg.obs_fps,
                forward_speed=cfg.robot_speed,
                timestep_gap=1 / cfg.obs_fps,
                robot=None,
            )

            h5_path = Path(f"data/raw_data/obj/multi_move/{worker_id}_episode_{all_index}.hdf5")
            h5_path.parent.mkdir(parents=True, exist_ok=True)

            save_move_obj_data_to_h5(
                output_data["obs"],
                walk_path=followed_path,
                h5_path=str(h5_path),
                item=item,
            )

            all_index += 1

        except Exception as e:
            print(f"[Worker {worker_id}] ERROR: {e}")

    simulator.close()

# ===============================
# 多进程入口函数
# ===============================

def worker_entry(args):
    worker_id, scene, items, yaml_file_path, num_gpus = args
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

    from habitat_for_sim.utils.goat import read_yaml
    cfg = read_yaml(yaml_file_path)

    data = load_json_group_by_scene(cfg.move_data_path)
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
