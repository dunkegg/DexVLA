import sys
import os
os.environ["HABITAT_SIM_EGL"] = "1"
os.environ["HABITAT_SIM_GPU_DEVICE_ID"] = "0"
from qwen2_vla.model_load_utils import load_model_for_eval
from torchvision import transforms
import pickle
import time
from data_utils.utils import set_seed
import random
from policy_heads import * 
from qwen2_vla.utils.image_processing_qwen2_vla import *  
import torch
import numpy as np
import h5py
import cv2
from evaluate.visualize_action import plot_actions, plot_obs
from collections import deque
import magnum as mn
from tqdm import tqdm
import habitat_sim
print(habitat_sim.__file__)
from habitat_for_sim.sim.habitat_utils import local2world_position_yaw, to_vec3, to_quat, shortest_angle_diff, load_humanoid
from habitat_sim.utils.common import quat_from_coeffs, quat_from_two_vectors , quat_from_angle_axis, quat_to_angle_axis 
from habitat_for_sim.utils.goat import read_yaml, extract_dict_from_folder, get_current_scene, process_episodes_and_goals, convert_to_scene_objects, find_scene_path, calculate_euclidean_distance
from habitat_for_sim.agent.path_generator import generate_path
from habitat_for_sim.utils.frontier_exploration import FrontierExploration
from scipy.spatial.transform import Rotation as R
# 将上级目录加入 Python 搜索路径

from habitat_for_sim.utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp

def process_obs(obs, states, stats):
    """
    obs: three cameras' images
    states: Tensor, robot states
    stats: mean, std of robot states and actions
    This function is used to get observations(images and robot states) in your robot environment.
    """

    cur_top = obs['top']
    assert np.max(cur_top) > 1, "All images must be 0-255."
    traj_rgb_np = np.array(cur_top) # sequential must align with constants.py
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    cur_state_np = pre_process(states, 'qpos', stats)
    cur_state = np.expand_dims(cur_state_np, axis=0)

    return traj_rgb_np, cur_state # images, states


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
    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang,9)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # top, left_wrist, right_wrist
        image_list = []
        for i, each in enumerate(image_data):
            ele = {}
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
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



class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self):
        self.history_obs = None
        self.qpos = None
        self.actions = None
        self.raw_lang = None
        self.policy = None
        self.policy_config = None
        self.post_process = None
        self.stats = None
        self.agent = None
        self.local_actions = None



    # def step(self, action):
    #     print("Execute action successfully!!!")

    def reset(self, agent):
        print("Reset to home position.")
        self.agent = agent
        self.history_obs = []
        self.chunk_size = 30
        self.query_frequency = 10
        self.action_queue = deque(maxlen=self.query_frequency)
        self.state = self.agent.get_state()
        self.height = self.state.position[1]
        self.qpos = np.array([0, 0, 0])
        
        

        raw_lang ="follow the human"
        self.instruction = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."

    def set_state(self, pos, rot):
        
        self.state.position = pos
        self.state.rotation = rot
        self.agent.set_state(self.state)

    def get_state(self):
        return self.agent.get_state()

    def set_obs(self,image):
        self.history_obs.append(image)
        # self.qpos = qpos

    def get_obs(self, n_frames):

        assert len(self.history_obs)>0
        cur_images = self.history_obs.copy()

        # 如果数量不足，则重复第一个元素
        if len(cur_images) < n_frames:
            padding = [cur_images[0]] * (n_frames - len(cur_images))
            cur_images = padding + cur_images  # 前面补齐

        else:
            cur_images = cur_images[-n_frames:]  # 多了就保留最后 n_frames 张

        obs = {
            'top': cur_images,
        }
        qpos = self.qpos
        # qpos = np.array([0,0,0])
        return obs, qpos
    
    def save_obs(self, time, human_position):
        assert len(self.history_obs)>0
        cur_image = self.history_obs[-1]

        human_position = np.array([human_position.x,human_position.y,human_position.z])
        human_position = human_position - self.agent.get_state().position
        human_position[0] = human_position[0]
        human_position[2] = -human_position[2]
        plot_obs(time, self.local_actions, self.post_process, "follow the human", cur_image,human_position)


    # def set_info(self,actions,raw_lang):
    #     self.actions = actions
    #     self.raw_lang = raw_lang
    # def get_info(self):
    #     return self.actions, self.raw_lang

    def set_policy(self,policy,policy_config):
        self.policy = policy
        self.policy_config = policy_config
        # self.policy.policy.eval()
        ## 4. load data stats(min,max,mean....) and define post_process####################################
        stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        if policy_config["action_head"].lower() == 'act':
            self.post_process = lambda a: a * self.stats ['action_std'] + self.stats ['action_mean']
        elif 'scale_dp_policy' in policy_config["action_head"]:
            self.post_process = lambda a: ((a + 1) / 2) * (self.stats ['action_max'] - self.stats ['action_min']) + self.stats ['action_min']
        #############################################################################################################

    def step(self, t):
        if len(self.action_queue) ==0:
            return
        habitat_action = self.action_queue.popleft()
        # raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
        ### 8. post process actions##########################################################
        # action = self.post_process(raw_action) !!wzj

        # print(f"Step, action is: {action}")
        #####################################################################################
        # print(f"after post_process action size: {action.shape}")
        # print(f'step {t}, pred action: {outputs}{action}')
        # if len(action.shape) == 2:
        #     action = action[0]
        ##### Execute ######################################################################
        # action = action.tolist()


        self.set_state(habitat_action[0], habitat_action[1])

        # action_info = self.step(action.tolist())


    def eval_bc(self,time_count):

        assert self.instruction  is not None, "raw lang is None!!!!!!"
        set_seed(0)
        rand_crop_resize = False
        #test

        all_actions = np.zeros((30, 3))         # 初始化为全0
        all_actions[:, 0] = np.linspace(0, 1, 30) 
        all_actions[:, 1] = np.linspace(0, 1, 30)  # 第1列填充从0到5平均分成30个数

        print(f"Plan, actions is: {all_actions[0:self.query_frequency]}")
        all_actions = torch.tensor(all_actions, dtype=torch.float32)
        self.local_actions = all_actions.to(dtype=torch.float32).cpu().numpy()

        #switch
        cur_state = self.get_state()
        cur_position = cur_state.position
        cur_rotation = cur_state.rotation
        cur_yaw,_ = quat_to_angle_axis(cur_rotation)
        
        local_actions = self.local_actions.copy()
        local_actions[:,0] = local_actions[:,0]
        local_actions[:,1] = -local_actions[:,1]
        height_dim = np.zeros((local_actions.shape[0], 1))
        # 拼接成 (30, 4)，在 dim=1 方向添加
        local_actions = np.insert(local_actions, 1, 0, axis=1)

        world_actions = local2world_position_yaw(local_actions, self.qpos, np.array([cur_position[0], cur_position[1], cur_position[2]]), cur_yaw)
        habitat_actions = []
        for i in range(len(world_actions)):
            pos = to_vec3([world_actions[i][0],world_actions[i][1],world_actions[i][2]])
            quat = quat_from_angle_axis(world_actions[i][3], np.array([0, 1, 0]))
            habitat_actions.append([pos, quat])
        

        self.action_queue.extend(
                habitat_actions[0:self.query_frequency])
        return
        image_list = []  # for visualization

        with torch.inference_mode():

            obs,states = self.get_obs()

            ### 5. Realize the function of get_obs###################
            traj_rgb_np, robot_state = process_obs(obs, states, self.stats)
            #########################################################
            image_list.append(traj_rgb_np)
            robot_state = torch.from_numpy(robot_state).float().cuda()
            
            ### 6. Augment the images##############################################################################################
            curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
            if rand_crop_resize:
                print('rand crop resize is used!')
                original_size = curr_image.shape[-2:]
                ratio = 0.95
                curr_image = curr_image[...,
                            int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                            int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                curr_image = curr_image.squeeze(0)
                resize_transform = transforms.Resize(original_size, antialias=True)
                curr_image = resize_transform(curr_image)
                curr_image = curr_image.unsqueeze(0)
            #######################################################################################################################

            ###7. Process inputs and predict actions############################################################################################
            batch = self.policy.process_batch_to_qwen2_vla(curr_image, robot_state, self.instruction )
            if policy_config['tinyvla']:
                all_actions, outputs = self.policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=self.policy.tokenizer)
            else:
                # from inspect import signature
                # print(signature(policy.policy.generate))
                all_actions, outputs = self.policy.policy.evaluate(**batch, is_eval=True, tokenizer=self.policy.tokenizer)

                # actions,raw_lang = self.get_info()
                # plot_actions(i,all_actions[0], actions, raw_lang, self.post_process, frames)
            self.actions = all_actions
            ####################################################################################################################################
            # clear previous actions
            while len(self.action_queue) > 0:
                self.action_queue.popleft()

            self.action_queue.extend(
                    torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:self.query_frequency])


        ####################################################################################


def follow(all_index, sim, robot, humanoid, controller, human_path, fps=10, forward_speed=0.7):
    
    output = {"obs":[], "follow_paths":[]}
    """
    path: [(mn.Vector3 pos, float yaw_rad), ...]
    """
    height_bias = 0
    keep_distance = 0.7
    for i in range(len(human_path)):
        pos, quat, yaw = human_path[i]            # 解包原 tuple
        new_pos  = mn.Vector3(pos.x, pos.y- height_bias, pos.z)
        human_path[i]  = (new_pos,quat, yaw)   


    observations=[]
    humanoid.base_pos = human_path[0][0]
    humanoid.base_rot = human_path[0][2]
    sim.step_physics(1.0 / fps)
    count = 0
    human_pos = []
    human_state = sim.agents[0].get_state()
    # follow_state.position = human_path[1][0]
    # follow_state.rotation = to_quat(human_path[1][1])
    follow_yaw = human_path[0][2]
    # sim.agents[0].set_state(follow_state)
    robot.set_state(human_path[0][0], to_quat(human_path[0][1]))
    #first 
    obs = sim.get_sensor_observations(0)
    robot.set_obs(obs)
    out_path = os.path.join("evaluate/plot_action2", f"{0}.png")  # 命名格式: frame_0000.png
    cv2.imwrite(out_path, obs["color_0_0"])


    move_dis = 0

    start_time = time.time()
    last_trigger_time = start_time
    last_action_time = start_time
    last_trigger_dis = 0
    last_action_dis = 0
    for k in range(1, len(human_path)):
        goal_pos, goal_quat,goal_yaw = human_path[k]

        # 位移向量
        start_pos = humanoid.base_pos
        seg_vec   = goal_pos - start_pos
        seg_len   = seg_vec.length()
        # move_dis += seg_len
        if seg_len < 1e-4:
            continue

        # 朝向增量
        start_yaw = humanoid.base_rot            # 当前 yaw (float)
        yaw_diff  = shortest_angle_diff(start_yaw, goal_yaw)

        # 行走分段
        direction = seg_vec.normalized()
        step_dist = forward_speed / fps
        n_steps   = int(np.ceil(seg_len / step_dist))
        
        for step in range(n_steps):
            # --- 1) 平移 ---
            humanoid.base_pos += direction * step_dist
            

            # --- 2) 线性插值 yaw ---
            frac = (step + 1) / n_steps        # 0→1
            humanoid.base_rot = start_yaw + yaw_diff * frac
            move_pos = humanoid.base_pos
            # --- 3) 步行动画，一帧 pose ---
            controller.calculate_walk_pose(seg_vec)   # 方向矢量给即可
            new_pose = controller.get_pose()

            new_joints = new_pose[:-16]
            new_pos_transform_base = new_pose[-16:]
            new_pos_transform_offset = new_pose[-32:-16]

            if np.array(new_pos_transform_offset).sum() != 0:
                vecs_base = [
                    mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                vecs_offset = [
                    mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                new_transform_offset = mn.Matrix4(*vecs_offset)
                new_transform_base = mn.Matrix4(*vecs_base)
                humanoid.set_joint_transform(
                    new_joints, new_transform_offset, new_transform_base
                )
                humanoid.base_pos = move_pos

            sim.step_physics(1.0 / fps)

            move_dis += step_dist
            now = time.time()
            if now - last_trigger_time >= 0.5: #robot act
                last_trigger_time = now
                obs = sim.get_sensor_observations(0)
                robot.set_obs(obs)
                time_count = round(now - start_time,2)
                robot.eval_bc(time_count)
                robot.save_obs(time_count,humanoid.base_pos)
                print("[trigger]", round(now - start_time, 2))
                

            # if now - last_action_time >= 1/10: #robot act
            #     last_action_time = now
            #     robot.step(round(now - start_time,2))
            #     print("[action]", round(now - start_time, 2))
            while now - last_action_time >= 1/10:
                last_action_time += 1/10
                robot.step(round(last_action_time - start_time, 2))
                print("[action]", round(now - start_time, 2))




if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 16
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        "model_path": "OUTPUT/qwen2_follow_20000/checkpoint-10000",
        #############################################################################
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
    }

    # fake env for debug
    agilex_bot = FakeRobotEnv()
    ######################################
    

    #### 3. Load DexVLA####################
    # policy = qwen2_vla_policy(policy_config)

    # agilex_bot.set_policy(policy, policy_config)



    folder = "/wangzejin/goat_bench/data/datasets/goat_bench/hm3d/v1/train/content" #wzjpath
    yaml_file_path = "habitat_for_sim/cfg/exp.yaml" #wzjpath
    
    # 初始化目标文件列表
    target_files = []   

    # 遍历文件夹并将相对路径添加到目标文件列表
    for root, dirs, files in os.walk(folder):
        for file in files:
            # 计算相对路径并加入列表
            relative_path = os.path.relpath(os.path.join(root, file), folder)
            target_files.append(relative_path)
    
    cfg = read_yaml(yaml_file_path)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    cfg.scenes_data_path = "/wangzejin/goat_bench/data/scene_datasets/hm3d/train" #wzjpath
    
    # data = extract_dict_from_folder(folder, target_files)

    all_index = 0
    for i in range(10):
    # for file_name, content in data.items():
        if all_index > 10:
            break

        # structured_data, filtered_episodes = process_episodes_and_goals(content)
        # episodes = convert_to_scene_objects(structured_data, filtered_episodes)
        
        # unique_episodes = {}
        # for ep in episodes:
        #     if ep["object_environment"] not in unique_episodes:
        #         unique_episodes[ep["object_environment"]] = ep

        # # 随机选择 5 个（如果少于 5 个，取全部）
        # random.seed(42)
        # episodes = random.sample(list(unique_episodes.values()), min(5, len(unique_episodes)))

        
        # scene = cfg.current_scene = get_current_scene(structured_data)
        
        
        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        # scene_mesh_dir = find_scene_path(cfg, cfg.current_scene)
        scene_mesh_dir = '/wangzejin/goat_bench/data/scene_datasets/hm3d/train/00529-W9YAR9qcuvN/W9YAR9qcuvN.basis.glb'
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": [0],
            "sensor_height": 1.5,
            "width": 640,
            "height": 480,
            "hfov": 120,
        }
        origin_sim_cfg = make_simple_cfg(sim_settings)

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_mesh_dir
        sim_cfg.load_semantic_mesh = False  # 禁用语义网格加载
        sim_cfg.enable_physics = False 
        sim_cfg.gpu_device_id = 0
        agent_cfg = habitat_sim.AgentConfiguration()
        radius = 0.4
        agent_cfg.radius = radius  # 设置 agent 的碰撞半径
        agent_cfg.height = 1.5  # 设置 agent 的高度
        num_sensors = 1
        angle = 2 * math.pi * i / num_sensors + math.pi / 2 # 计算每个传感器的角度：等夹角环绕360度
        #angle = 0 时 为正前方摄像头
        
        # RGB传感器配置
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = f"color_{0}_{0}"  # 每个传感器的唯一ID
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]

        # 四元数到欧拉角的转换
        rotation = quat_from_angle_axis(angle - math.pi / 2, np.array([0, 1, 0]))  # 计算四元数
        euler_angles = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_euler('xyz', degrees=False)  # 转换为欧拉角
        
        rgb_sensor_spec.position = [
            radius * math.cos(angle),  # x坐标
            sim_settings["sensor_height"],  # y坐标（高度）
            - radius * math.sin(angle)  # z坐标
        ]
        #print(f"position of sensor {i}:", rgb_sensor_spec.position)
        rgb_sensor_spec.orientation = euler_angles  # 设置欧拉角
        rgb_sensor_spec.hfov = sim_settings["hfov"]
        sensor_specs = []
        sensor_specs.append(rgb_sensor_spec)
        agent_cfg.sensor_specifications = sensor_specs

        all_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        simulator = habitat_sim.Simulator(all_cfg)

        # simulator = habitat_sim.Simulator(origin_sim_cfg)
        agilex_bot.reset(simulator.agents[0])
        # 从 sim_cfg 中获取 agent 配置
        agent_cfg = all_cfg.agents[0]  # 获取默认代理的配置
        # 获取 NavMeshSettings 对象
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.agent_radius = agent_cfg.radius             # 设置 agent 碰撞半径
        navmesh_settings.agent_height = agent_cfg.height             # 设置 agent 碰撞高度
        navmesh_settings.agent_max_climb = 1          # 设置最大爬升高度
        navmesh_settings.agent_max_slope = 45.0         # 设置最大坡度角度（单位：度）

        # 重新生成导航网格
        navmesh_success = simulator.recompute_navmesh(simulator.pathfinder, navmesh_settings)

        # 验证导航网格是否成功生成
        if not navmesh_success or not simulator.pathfinder.is_loaded:
            raise RuntimeError("Navmesh recomputation failed. Cannot proceed with pathfinding.")

        semantic_scene = simulator.semantic_scene
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        if not simulator.pathfinder.is_loaded:
            print("Failed to load or generate navmesh.")
            raise RuntimeError("Failed to load or generate navmesh.")   

        humanoid, controller = load_humanoid(simulator)

        print("begin")
        for episode_id, obj_data in enumerate(tqdm(episodes)):
            #print("obj_data:", obj_data)
            episode_data = obj_data
            
            # 设置起始位置和旋转
            start_position = obj_data.start_position
            start_rotation = obj_data.start_rotation
            distance = obj_data.info['euclidean_distance']
            goal_position = obj_data.goal["position"]

            start_normal = pos_habitat_to_normal(start_position)
            start_floor_height = start_normal[-1]



            if goal_position is None or start_position is None:
                continue
            # 使用 ShortestPath 对象生成避免穿墙的最短路径
            shortest_path = habitat_sim.ShortestPath()
            shortest_path.requested_start = start_position
            shortest_path.requested_end = goal_position

            # 查找最短路径 # 模拟前沿探索过程
            if pathfinder.find_path(shortest_path):
                
                # 检查起始点和目标点是否在同一楼层 (高度差小于1m)
                start_floor_height = start_position[1]  # y 值代表高度
                goal_floor_height = goal_position[1]
                if abs(start_floor_height - goal_floor_height) > 1:
                    print(f"Skipping episode due to height difference: {start_floor_height} vs {goal_floor_height}")
                    continue
                path = shortest_path.points
                all_distance = 0
                for i in range(len(path)-1):
                    distance = calculate_euclidean_distance(path[i], path[i+1])
                    all_distance+=distance
                
                if all_distance < 10:
                    print(f"Skipping episode due to short distance: {all_distance}m")
                    continue
                

                # 检查路径是否跨楼层 (高度差小于1m)
                floor_heights = [point[1] for point in path]  # 获取所有路径点的高度
                if max(floor_heights) - min(floor_heights) > 1:
                    print("Skipping episode due to multi-floor path")
                    continue
                
                # 初始化探索类
                if not cfg.shortest_path:

                    num_frontiers = random.randint(1, 4)
                    #print("original path:",path)
                    explorer = FrontierExploration(simulator)
                    explorer.explore_until_target(
                                start_position = start_position,
                                target_position = goal_position,
                                num_frontiers = num_frontiers)
                    path = explorer.trail
                #print("frontier_exploration path:", path)
                

                #轨迹优化流水线
                new_path = generate_path(path, pathfinder, visualize=False)
                
                #print(path[0])
                #print(path)
            else:
                print("No valid path found:", obj_data["object_category"])
                continue 
            
            # new_path = convert_path(path)


            follow(episode_id, simulator,agilex_bot, humanoid, controller, new_path,fps=10, forward_speed=0.1)



            print("done")


