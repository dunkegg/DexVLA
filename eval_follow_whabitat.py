import sys
import os
from qwen2_vla.model_load_utils import load_model_for_eval
import time

import random
from policy_heads import * 
from qwen2_vla.utils.image_processing_qwen2_vla import *  
import torch
import numpy as np
import h5py
import cv2
import json
import magnum as mn
from tqdm import tqdm
import habitat_sim
print(habitat_sim.__file__)

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
from evaluate_dexvln.robot import FakeRobotEnv


def time_ms():
    return time.time_ns() // 1_000_000

class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.multimodal_processor, self.context_len = load_model_for_eval(model_path=model_path,
                                                                                                    model_base=model_base, policy_config=policy_config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})

        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)
    def datastruct_droid2qwen2vla(self, raw_lang,len_image):

        messages = [
            {
                "role": "user",
                "content": [
                ],
            },
            # {"role": "assistant", "content": f''},
        ]
        for i in range(len_image):
            messages[0]['content'].append({
                "type": "image",
                "image": None,
            })
        messages[0]['content'].append({
            "type": "text",
            "text": raw_lang,
        })
        # messages[0]['content'][-1]['text'] = raw_lang

        return messages
    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang,n_frames):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang,n_frames)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # top, left_wrist, right_wrist
        image_list = []
        for i, each in enumerate(image_data):
            ele = {}
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            if each.mode == 'RGBA':
                each = each.convert('RGB')  # 去掉 alpha 通道

            ele['image'] = each
            ele['resized_height'] = 240
            ele['resized_width'] = 320

            image_list.append(torch.from_numpy(np.array(each)))
        # image_data = image_data / 255.0
        image_data = image_list
        ######################
        video_inputs = [image_list]
        # image_data = None
        video_inputs=None
        ######################
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict





if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 16
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        "model_path": "OUTPUT/single_follow_normal/checkpoint-20000",
        #############################################################################
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
    }

    # fake env for debug
    policy = qwen2_vla_policy(policy_config)
    agilex_bot = FakeRobotEnv(policy_config, policy)
    ######################################
    

    yaml_file_path = "habitat_for_sim/cfg/exp.yaml"
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
    
    
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    
    data = extract_dict_from_folder(json_data, target_files)
    
    max_episodes = cfg.max_episodes

    all_index = 0
    episodes_count = 0
    for file_name, content in data.items():
        if episodes_count > max_episodes:
            break
        

        print(f"Processing {file_name}:")
        structured_data,  filtered_episodes = process_episodes_and_goals(content)
        episodes = convert_to_scene_objects(structured_data, filtered_episodes)
                
        cfg.current_scene = get_current_scene(structured_data)
        
        
        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass

        simulator = load_simulator(cfg)
        agilex_bot.reset(simulator.agents[0],n_frames=6)
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
        
        # 原主目标人
        description = id_dict[humanoid_name]["description"]
        target_humanoid = AgentHumanoid(simulator,base_pos=mn.Vector3(0, 0.083, 0), base_yaw = 0,name = humanoid_name,description = description, is_target=True)
        
        all_interfering_humanoids = []
        if cfg.multi_humanoids:
            for idx in range(3):
                # break
                # max_humanoids[idx].reset(name = get_humanoid_id(humanoid_name))
                interferer_name = get_humanoid_id(id_dict, name_exception = humanoid_name)
                interferer_description = id_dict[humanoid_name]["description"]
                interferer = AgentHumanoid(simulator, base_pos=mn.Vector3(0, 0.083, 0), base_yaw = 0, name = interferer_name, description = interferer_description, is_target=False)
                all_interfering_humanoids.append(interferer)


        print("begin")
        for episode_id, episode_data in enumerate(tqdm(episodes)):

            if episodes_count > max_episodes:
                break

            human_fps = 5
            human_speed = 0.7
            followed_path = generate_path_from_scene(episode_data, pathfinder, human_fps, human_speed)
            if followed_path is None:
                continue
            
            #
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
                

            output_data = walk_along_path_multi(
                all_index=all_index,
                sim=simulator,
                humanoid_agent=target_humanoid,
                human_path=followed_path,
                fps=10,
                timestep_gap = 1/human_fps, 
                interfering_humanoids=interfering_humanoids,
                robot = agilex_bot
            )


            print(f"Case {all_index}, {humanoid_name} Done, Already has {episodes_count} cases")
            all_index+=1





            print("done")


