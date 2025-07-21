import os
import pickle
from torchvision import transforms

from habitat_for_sim.sim.habitat_utils import local2world_position_yaw, to_vec3, to_quat, shortest_angle_diff, load_humanoid
from habitat_sim.utils.common import quat_from_coeffs, quat_from_two_vectors , quat_from_angle_axis, quat_to_angle_axis 
from data_utils.utils import set_seed
import torch
import numpy as np
from evaluate.visualize_action import plot_actions, plot_obs
from collections import deque
import imageio

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



class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self, policy_config, policy):
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

        self.set_policy(policy, policy_config)
        

    # def step(self, action):
    #     print("Execute action successfully!!!")

    def reset(self, agent, n_frames):
        print("Reset to home position.")
        self.agent = agent
        self.history_obs = []
        self.chunk_size = 30
        self.query_frequency = 10
        self.action_queue = deque(maxlen=self.query_frequency)
        self.state = self.agent.get_state()
        self.height = self.state.position[1]
        self.qpos = np.array([0, 0, 0])
        self.n_frames = n_frames
        
        

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
        human_position = human_position - self.agent.get_state().position
        human_position[0] = human_position[0]
        human_position[2] = -human_position[2]
        img_np = plot_obs(time, self.local_actions, "follow the human", cur_image,human_position)
        imageio.imwrite(f'eval_plot/{time}.png', img_np)


    # def set_info(self,actions,raw_lang):
    #     self.actions = actions
    #     self.raw_lang = raw_lang
    # def get_info(self):
    #     return self.actions, self.raw_lang

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


    def eval_bc(self):

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


            ####################################################################################
