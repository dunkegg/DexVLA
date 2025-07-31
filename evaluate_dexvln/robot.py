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
from evaluate.visualize_action import plot_actions, plot_obs
from collections import deque
import imageio
from PIL import Image


from qwen2_vla.model_load_utils import load_model_for_eval
from policy_heads import * 
from qwen2_vla.utils.image_processing_qwen2_vla import *  


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
        

    # def step(self, action):
    #     print("Execute action successfully!!!")

    def reset(self, agent, n_frames):
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

        raw_lang ="follow the human"
        self.instruction = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."

    def set_episode_id(self, episode_id):
        self.episode_id = episode_id

    def set_state(self, pos, rot):
        
        self.state.position = pos
        self.state.rotation = rot
        self.agent.set_state(self.state)

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
        local_human_position = world2local_target(human_position - self.agent.get_state().position, habitat_quat_to_magnum(self.agent.get_state().rotation), type=1)
        # human_position[0] = human_position[0]
        # human_position[2] = -human_position[2]
        plot_world = np.concatenate([self.world_actions[:, :1], self.world_actions[:, 2:]], axis=1)
        world_img_np = plot_obs(time, plot_world, "follow the human", cur_image,human_position)
        local_img_np = plot_obs(time, self.local_actions, "follow the human", cur_image,local_human_position)
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
            while len(self.action_queue) > 18:
                self.step_actions.append(self.action_queue.popleft())

        else:
            new_actions = []
            while len(self.action_queue) > 18:
                new_actions.append(self.action_queue.popleft())
            
            origin_future_steps = self.step_actions[self.step_idx:self.step_idx+comp_size]
            new_future_steps = new_actions[0:comp_size]
            changed = False
            if changed:
                self.step_actions = new_actions
                self.step_idx = 0
            else:
                move_dis = 0
                for i in range(self.step_idx, len(self.step_actions)):
                    if move_dis > distance:
                        self.step_idx = i
                        break

                    cur_action = self.step_actions[i]
                    next_action = self.step_actions[i+1]
                    seg_vec = next_action[0] - cur_action[0]

                    seg_len = seg_vec.length()
                    if seg_len <1e-4:
                        continue
                    move_dis += seg_len

                    direction = seg_vec.normalized()
                    orientation = direction / np.linalg.norm(direction) 
                    quaternion = direction_to_combined_quaternion(orientation)
                    self.set_state(cur_action[0], quaternion)

            


        
        

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
            cur_quat = cur_state.rotation
            cur_yaw,_ = quat_to_angle_axis(cur_quat)
            
            local_actions = self.local_actions.copy()
            # local_actions[:,0] = -local_actions[:,0]
            # local_actions[:,1] = local_actions[:,1]
            height_dim = np.zeros((local_actions.shape[0], 1))
            # 拼接成 (30, 4)，在 dim=1 方向添加
            local_actions = np.insert(local_actions, 1, 0, axis=1)
            
            world_actions = local2world(local_actions, np.array([cur_position[0], cur_position[1], cur_position[2]]), habitat_quat_to_magnum(cur_quat),cur_yaw, type=1)
            self.world_actions = world_actions
            habitat_actions = []
            for i in range(len(world_actions)):
                pos = to_vec3([world_actions[i][0],world_actions[i][1],world_actions[i][2]])
                quat = quat_from_angle_axis(world_actions[i][3], np.array([0, 1, 0]))
                habitat_actions.append([pos, quat])
            

            self.action_queue.extend(
                    habitat_actions[0:self.query_frequency])


            ####################################################################################
