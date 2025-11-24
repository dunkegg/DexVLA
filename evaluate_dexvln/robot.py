import os
import pickle
from torchvision import transforms
from habitat_for_sim.agent.path_generator import direction_to_combined_quaternion
from habitat_for_sim.sim.habitat_utils import local2world,habitat_quat_to_magnum ,to_vec3, to_quat, shortest_angle_diff, load_humanoid
from process_data.process_raw_h5 import world2local_target
from habitat_sim.utils.common import quat_from_coeffs, quat_from_two_vectors , quat_from_angle_axis, quat_to_angle_axis 
from data_utils.utils import set_seed
import torch
import numpy as np
from evaluate.visualize_action import plot_actions, plot_obs, plot_ctrl
from collections import deque
import imageio
from PIL import Image

from evaluate_dexvln.controller import TrajectoryFollower

from qwen2_vla.model_load_utils import load_model_for_eval
from policy_heads import * 
from qwen2_vla.utils.image_processing_qwen2_vla import *  

import magnum as mn
import quaternion as qt 

def adjust_yaw_sign(follow_world_actions, w_yaw, pos_cur, pos_goal, check_k=3):
    """
    修正 follow_world_actions 的 yaw 符号

    follow_world_actions: ndarray, shape (N, 3) or (N, 4)，其中 [:,2] 是 yaw
    w_yaw: 当前机器人 yaw (float)
    pos_cur: [x, y] 当前坐标
    pos_goal: [x, y] 目标坐标
    check_k: 用轨迹前几个点来计算方向
    """
    yaws = follow_world_actions[:, 2].copy()

    yaw0 = yaws[0]

    # ---------------------------
    # 1. 初步判断 yaw 是否对齐
    if np.isclose(yaw0, w_yaw, atol=0.5):  # yaw 差不多（阈值可调）
        return follow_world_actions  # 不变
    elif np.isclose(yaw0 + w_yaw, 0, atol=0.5):
        follow_world_actions[:, 2] = -yaws
        return follow_world_actions
    else:
        # 初始 yaw 明显不对，进入进一步检查
        pass

    # ---------------------------
    # 2. 方向检查（用当前位置 -> 目标向量 和 轨迹向量）
    vec_goal = np.array(pos_goal) - np.array(pos_cur)
    if np.linalg.norm(vec_goal) < 1e-6:
        return follow_world_actions  # 当前位置和目标一样，直接返回

    vec_traj = follow_world_actions[min(check_k, len(follow_world_actions)-1), :2] - follow_world_actions[0, :2]

    # 归一化
    vec_goal /= np.linalg.norm(vec_goal)
    vec_traj /= np.linalg.norm(vec_traj)

    cos_angle = np.dot(vec_goal, vec_traj)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    if angle <= np.pi / 2:  # 夹角小于等于90度 → 不取负
        return follow_world_actions
    else:
        follow_world_actions[:, 2] = -yaws
        return follow_world_actions

def compute_yaw_from_xy(path_xyyaw, cur_yaw):
    xy = path_xyyaw[:, :2]  # 取 x, y
    yaw_list = []
    for i in range(len(xy)):
        if i < len(xy) - 1:
            dx = xy[i+1, 0] - xy[i, 0]
            dy = xy[i+1, 1] - xy[i, 1]
        else:  # 最后一个点用前一个点的方向
            dx = xy[i, 0] - xy[i-1, 0]
            dy = xy[i, 1] - xy[i-1, 1]
        
        seg_vec = mn.Vector3(dx, 0, dy)
        direction = seg_vec.normalized()
        orientation = direction / np.linalg.norm(direction) 
        q_array = direction_to_combined_quaternion(orientation)
        # quat = qt.quaternion(q_array[0],q_array[1], q_array[2], q_array[3])
        quat = qt.quaternion(q_array[3],q_array[0], q_array[1], q_array[2])
        yaw, _ = quat_to_angle_axis(quat) 
        # yaw += math.pi/2
        # yaw = np.arctan2(dy, dx)  # -pi ~ pi
        yaw_list.append(yaw)

    yaw_diff = yaw_list[0] + cur_yaw
    yaw_array = np.array(yaw_list)
    yaw_array = yaw_array - yaw_diff

    return np.hstack([xy, yaw_array[:, None]])  # 拼回 [x, y, yaw]

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
    
def smooth_quat(actions, window_size=3):
    if len(actions) == 0:
        return []
    height = actions[0][0][1]
    yaw_smooth = []
    for pos, quat in actions:
        yaw, _ = quat_to_angle_axis(quat)
        yaw_smooth.append((np.array(pos), yaw))
    new_smooth = np.array([np.array([pos[0], pos[2], yaw]) for pos, yaw in yaw_smooth])
    new_smooth = smooth_yaw(new_smooth, window_size)
    quat_smooth = []
    for s in new_smooth:
        quat_smooth.append([to_vec3([s[0],height ,s[1]]), quat_from_angle_axis(s[2], np.array([0, 1, 0]))])
    return quat_smooth

def smooth_yaw(actions, window_size=3):
    if len(actions) < window_size:
        return actions
    
    yaw_actions = []
    for action in actions:
        yaw_actions.append((action[:2], action[2]))
    
    smoothed_actions = []
    for i in range(len(yaw_actions)):
        start = max(0, i - window_size // 2)
        end = min(len(yaw_actions), i + window_size // 2 + 1)
        window = yaw_actions[start:end]

        avg_pos = sum([a[0] for a in window]) / len(window)
        avg_yaw = sum([a[1] for a in window]) / len(window)
        smoothed_actions.append((avg_pos, avg_yaw))
    
    final = np.array([np.array([pos[0], pos[1], yaw]) for pos, yaw in smoothed_actions])
    return final


def adjust_yaw_sign2(follow_world_actions, yaw_cmd):
    follow_world_actions = np.delete(follow_world_actions, 1, axis=1)
    """
    修正 follow_world_actions 的 yaw 符号

    follow_world_actions: ndarray, shape (N, 3) or (N, 4)，其中 [:,2] 是 yaw
    w_yaw: 当前机器人 yaw (float)
    pos_cur: [x, y] 当前坐标
    pos_goal: [x, y] 目标坐标
    check_k: 用轨迹前几个点来计算方向
    """

    vec_traj = follow_world_actions[-1, :2] - follow_world_actions[0, :2]
    if np.linalg.norm(vec_traj) < 1e-6:
        return yaw_cmd  # 轨迹几乎没动，直接返回
    # yaw_cmd 转方向向量（注意加 yaw_bias）
    yaw_bias = math.pi / 2
    vec_1 = np.array([np.cos(yaw_cmd + yaw_bias), np.sin(yaw_cmd + yaw_bias)])
    vec_2 = np.array([np.cos(-yaw_cmd + yaw_bias), np.sin(-yaw_cmd + yaw_bias)])

    # 计算两个方向与轨迹的相似度（点积）
    dot1 = np.dot(vec_1, vec_traj)
    dot2 = np.dot(vec_2, vec_traj)

    # 选择更接近轨迹方向的 yaw
    if dot1 >= dot2:
        return yaw_cmd  # 保持不变
    else:
        return -yaw_cmd  # 翻转符号

class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self, policy_config, policy, plot_dir = "eval_plot"):
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
        self.world_actions = None

        self.set_policy(policy, policy_config)
        self.episode_id = None
        self.plot_dir = plot_dir
        
        self.smooth_window_size = 6

        self.follower = None
        self.type = 1
    # def step(self, action):
    #     print("Execute action successfully!!!")

    def reset(self, agent, n_frames, human_description):
        print("Reset to home position.")
        self.agent = agent
        self.history_obs = []
        self.chunk_size = 30
        self.query_frequency = 30
        self.action_queue = deque(maxlen=self.query_frequency)
        self.state = self.agent.get_state()
        self.height = self.state.position[1]
        self.qpos = np.array([0, 0, 0])
        self.n_frames = n_frames
        self.world_actions = []
        self.local_actions = []
        self.step_actions = []
        self.step_idx = 0

        # self.plot_dir = self.plot_dir + f"{n_frames}"

        self.raw_lang ="follow the human"
        if human_description:
            self.raw_lang = "follow :" + human_description
        self.instruction = f"Your task is: {self.raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."

    def set_episode_id(self, episode_id):
        self.episode_id = episode_id

    def set_state(self, pos, rot):
        
        self.state.position = pos
        self.state.rotation = rot
        self.agent.set_state(self.state)

    def set_world_pos(self, x,y,yaw, height):
        self.w_x = x
        self.w_y = y
        self.w_yaw = yaw
        self.w_height = height
        self.origin_pos = [self.w_x,self.w_y,self.w_yaw]

    def get_state(self):
        return self.agent.get_state()
    def get_observations(self):
        return self.history_obs

    def set_obs(self,image,time,save=False):
        # 如果是 numpy 数组，先转为 PIL.Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 确保没有 alpha 通道
        if image.mode == 'RGBA':
            image_rgb = image.convert('RGB')

        self.history_obs.append(image)
        if save:
            plot_dir = os.path.join(self.plot_dir, f"episode_{self.episode_id}","sample")
            os.makedirs(plot_dir, exist_ok=True)
            imageio.imwrite(f'{plot_dir}/{round(time, 1)}.png', image_rgb)
            # imageio.imwrite(f'{plot_dir}/{round(time, 1)}_a.png', image)
        # self.qpos = qpos

    def get_obs(self):
        n_frames = self.n_frames
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
        local_human_position = world2local_target(human_position - self.agent.get_state().position, habitat_quat_to_magnum(self.agent.get_state().rotation), type=self.type)
        # human_position[0] = human_position[0]
        # human_position[2] = -human_position[2]
        plot_world = np.concatenate([self.world_actions[:, :1], self.world_actions[:, 2:]], axis=1)
        
        #smooth
        # world_actions_xy = np.concatenate([self.world_actions[:, :1], self.world_actions[:, 2:]], axis=1)
        # world_actions_xy_smooth = smooth_np(world_actions_xy) 
        
        local_actions_smooth = smooth_yaw(self.local_actions, self.smooth_window_size)

        # world_img_np = plot_obs(time, plot_world, "follow the human", cur_image,human_position)
        local_img_np = plot_obs(time, self.local_actions, self.raw_lang, cur_image,local_human_position, smooth_actions=local_actions_smooth)

        plot_dir = os.path.join(self.plot_dir, f"episode_{self.episode_id}")
        os.makedirs(plot_dir, exist_ok=True)
        imageio.imwrite(f'{plot_dir}/{round(time, 1)}_local.png', local_img_np)
        # imageio.imwrite(f'{plot_dir}/{round(time, 1)}_world.png', world_img_np)

    def set_policy(self,policy,policy_config):
        self.policy = policy
        self.policy_config = policy_config
        
        ## 4. load data stats(min,max,mean....) and define post_process####################################
        stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        if policy_config["action_head"].lower() == 'act':
            self.post_process = lambda a: a * self.stats ['action_std'] + self.stats ['action_mean']
        elif 'scale_dp_policy' in policy_config["action_head"]:
            self.post_process = lambda a: ((a + 1) / 2) * (self.stats ['action_max'] - self.stats ['action_min']) + self.stats ['action_min']
        #############################################################################################################

    # def step(self,t, originla_quat = True):
    #     
    #     if len(self.action_queue) ==0:
    #         return
    #     cur_action = self.action_queue.popleft()
    #     last_action = cur_action
    #     while len(self.action_queue)>0:
    #         last_action = cur_action
    #         cur_action = self.action_queue.popleft()
    #         seg_vec = cur_action[0] - last_action[0]
    #         direction = seg_vec.normalized()
    #         orientation = direction / np.linalg.norm(direction) 
    #         quaternion = direction_to_combined_quaternion(orientation)
    #         angle_rad = np.arctan2(direction[2],direction[0])
    #         angle_deg = np.degrees(angle_rad)
    #         # print(f"cal direction is {angle_deg}")

    #     if originla_quat:
    #         self.set_state(cur_action[0], cur_action[1])
    #     else:
    #         self.set_state(cur_action[0], quaternion)
    def step(self,t, originla_quat = True):
        from habitat_for_sim.agent.path_generator import direction_to_combined_quaternion
        if len(self.action_queue) <=1:
            return
        last_action = self.action_queue.popleft()
        cur_action = self.action_queue.popleft()
        seg_vec = cur_action[0] - last_action[0]
        direction = seg_vec.normalized()
        orientation = direction / np.linalg.norm(direction) 
        quaternion = direction_to_combined_quaternion(orientation)
        if originla_quat:
            self.set_state(cur_action[0], cur_action[1])
        else:
            self.set_state(cur_action[0], quaternion)

    def compare_step(self, comp_size, distance):
        
        if self.step_idx + comp_size > len(self.step_actions) - 1:
            self.step_actions = []
            self.step_idx = 0
            while len(self.action_queue) > 0:
                self.step_actions.append(self.action_queue.popleft())

            self.step_actions = smooth_quat(self.step_actions, self.smooth_window_size)


        new_actions = []
        while len(self.action_queue) > 0:
            new_actions.append(self.action_queue.popleft())
        #smooth
        new_actions = smooth_quat(new_actions, self.smooth_window_size)
        origin_future_steps = self.step_actions[self.step_idx:self.step_idx+comp_size]
        # origin_smooth = smooth_quat(origin_future_steps, self.smooth_window_size)

        new_future_steps = new_actions[0:comp_size]
        # new_smooth = smooth_quat(new_future_steps, self.smooth_window_size)
        changed = False
        # for (p1, y1), (p2, y2) in zip(origin_smooth, new_smooth):
        #     pos_diff = np.linalg.norm(p1 - p2)
        #     yaw_diff = abs(shortest_angle_diff(y1, y2))
        #     if pos_diff > 0.2 or yaw_diff > 0.2:
        #         changed = True
        #         break
        # changed = False
        if changed:
            self.step_actions = new_actions
            self.step_idx = 0

        move_dis = 0

        first_point = self.step_actions[0]
        last_point = self.step_actions[-1]
        seg_vec = last_point[0] - first_point[0]
        seg_len = seg_vec.length()
        if seg_len <0.2:
            self.step_idx = 0
            self.step_actions = []
            return

        if seg_len <0.4:
            self.step_idx = 0
            self.step_actions = []
            direction = seg_vec.normalized()
            orientation = direction / np.linalg.norm(direction) 
            quaternion = direction_to_combined_quaternion(orientation)
            self.set_state(last_point[0], quaternion)

        for i in range(self.step_idx+1, len(self.step_actions)):
            if move_dis > distance:
                break

            cur_action = self.step_actions[i-1]
            next_action = self.step_actions[i]
            seg_vec = next_action[0] - cur_action[0]

            seg_len = seg_vec.length()
            if seg_len <1e-4:
                continue
            move_dis += seg_len

            direction = seg_vec.normalized()
            orientation = direction / np.linalg.norm(direction) 
            quaternion = direction_to_combined_quaternion(orientation)
            self.set_state(next_action[0], quaternion)
            self.step_idx = i


    def ctrl_step(self, now_time, followed_position):
        if not self.follower or len(self.world_actions)==0:
            return
        
        agent_pos = [self.w_x,self.w_y,self.w_yaw]
        x_cmd, y_cmd, yaw_cmd, idx = self.follower.step(now_time, self.w_x, self.w_y, self.w_yaw)
        # yaw_cmd = -yaw_cmd
        yaw_cmd = adjust_yaw_sign2(self.world_actions, yaw_cmd)

        pid_pos = [x_cmd, y_cmd, yaw_cmd]
        followed_pos = [followed_position.x,followed_position.z]
        assert len(self.history_obs)>0
        cur_image = self.history_obs[-1]
        ctrl_img_np = plot_ctrl(now_time, self.world_actions, pid_pos, agent_pos, followed_pos, self.origin_pos,cur_image)
        plot_dir = os.path.join(self.plot_dir, f"episode_{self.episode_id}","PID")
        os.makedirs(plot_dir, exist_ok=True)
        imageio.imwrite(f'{plot_dir}/{round(now_time, 1)}.png', ctrl_img_np)
        print(f"now time: {now_time}, yaw cmd: {yaw_cmd},  traj yaw: {self.world_actions[idx][3]}, idx:{idx},  cur_yaw: {self.w_yaw }")
        print(f"whole traj: {self.world_actions[:,3] }")
        print("--------------------------------------------------------")


        self.w_x = x_cmd
        self.w_y = y_cmd
        self.w_yaw = yaw_cmd
        pos = to_vec3([self.w_x, self.w_height, self.w_y])
        quat = quat_from_angle_axis(self.w_yaw , np.array([0, 1, 0]))
        self.set_state(pos, quat)
        

    def eval_bc(self, now_time, followed_position):

        assert self.instruction  is not None, "raw lang is None!!!!!!"
        set_seed(0)
        rand_crop_resize = False
        #test

        # all_actions = np.zeros((30, 3))         # 初始化为全0
        # all_actions[:, 0] = np.linspace(0, 1, 30) 
        # all_actions[:, 1] = np.linspace(0, 1, 30)  # 第1列填充从0到5平均分成30个数

        # print(f"Plan, actions is: {all_actions[0:self.query_frequency]}")
        # all_actions = torch.tensor(all_actions, dtype=torch.float32)
        # self.local_actions = all_actions.to(dtype=torch.float32).cpu().numpy()

        self.policy.policy.eval()
        
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
            batch = self.policy.process_batch_to_qwen2_vla(curr_image, robot_state, self.instruction,self.n_frames)
            if self.policy_config['tinyvla']:
                all_actions, outputs = self.policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=self.policy.tokenizer)
            else:
                # from inspect import signature
                # print(signature(policy.policy.generate))
                all_actions, outputs = self.policy.policy.evaluate(**batch, is_eval=True, tokenizer=self.policy.tokenizer)
                all_actions = all_actions.squeeze(0)  #
                all_actions = all_actions.to(dtype=torch.float32).cpu().numpy()
                all_actions = np.array([self.post_process(raw_action) for raw_action in all_actions])
                # actions,raw_lang = self.get_info()
                # plot_actions(i,all_actions[0], actions, raw_lang, self.post_process, frames)
            self.local_actions = all_actions
            ####################################################################################################################################
            # clear previous actions
            while len(self.action_queue) > 0:
                self.action_queue.popleft()

            #switch
            cur_state = self.get_state()
            cur_position = cur_state.position
            cur_quat = cur_state.rotation
            cur_yaw,_ = quat_to_angle_axis(cur_quat)
            
            local_actions = self.local_actions.copy()
            # local_actions[:,0] = -local_actions[:,0]
            # local_actions[:,1] = local_actions[:,1]
            height_dim = np.zeros((local_actions.shape[0], 1))
            # 拼接成 (30, 4)，在 dim=1 方向添加

            local_actions = smooth_yaw(local_actions, self.smooth_window_size)

            local_actions = np.insert(local_actions, 1, 0, axis=1)
            
            raw_yaw_world_actions = local2world(local_actions, np.array([cur_position[0], cur_position[1], cur_position[2]]), habitat_quat_to_magnum(cur_quat),self.w_yaw, type=self.type)
            
            raw_yaw_world_actions = np.delete(raw_yaw_world_actions, 1, axis=1)
            world_actions = compute_yaw_from_xy(raw_yaw_world_actions, self.w_yaw)
            world_actions = np.insert(world_actions, 1, 0, axis=1)
            self.world_actions = world_actions

            # total_time = path_length / v_des
            

            habitat_actions = []
            for i in range(len(world_actions)):
                pos = to_vec3([world_actions[i][0],world_actions[i][1],world_actions[i][2]])
                quat = quat_from_angle_axis(world_actions[i][3], np.array([0, 1, 0]))
                habitat_actions.append([pos, quat])

            first_point = to_vec3([world_actions[0][0],world_actions[0][1],world_actions[0][2]])
            last_point = to_vec3([world_actions[-1][0],world_actions[-1][1],world_actions[-1][2]])
            seg_vec = last_point - first_point
            seg_len = seg_vec.length()
            if seg_len <0.1:
                follow_world_actions = [[self.w_x, self.w_y, self.w_yaw]]
            else:
                follow_world_actions = np.delete(world_actions, 1, axis=1)
                # follow_world_actions = adjust_yaw_sign(follow_world_actions, self.w_yaw, [self.w_x, self.w_y],  [followed_position.x,followed_position.z])

            self.follower = TrajectoryFollower(follow_world_actions, total_time=0.75, 
                                               kp_xy=1.0, ki_xy=0.0, kd_xy=0.0,
                                               kp_yaw=0.5, ki_yaw=0.0, kd_yaw=0.0)
            
            self.follower.reset(now_time)
            self.world_actions = np.insert(follow_world_actions, 1, self.height, axis=1)
            

            self.action_queue.extend(
                    habitat_actions[0:self.query_frequency])


            ####################################################################################
