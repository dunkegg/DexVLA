import os
import pickle
from torchvision import transforms

from data_utils.utils import set_seed
import torch
import numpy as np
from evaluate.visualize_action import plot_actions, plot_obs
from collections import deque
import imageio
from PIL import Image
import time
import cv2

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



class RawRobotEnv():
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

    def reset(self, n_frames, human_description):
        print("Reset to home position.")

        self.history_obs = []
        self.chunk_size = 30
        self.query_frequency = 30
        self.action_queue = deque(maxlen=self.query_frequency)
        self.qpos = np.array([0, 0, 0])
        self.n_frames = n_frames
        self.world_actions = []
        self.local_actions = []
        self.step_actions = []
        self.step_idx = 0

        # self.plot_dir = self.plot_dir + f"{n_frames}"

        raw_lang ="follow the human"
        if human_description:
            raw_lang = "follow :" + human_description
        self.instruction = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."

    def set_episode_id(self, episode_id):
        self.episode_id = episode_id

    def set_state(self, pos, rot):
        
        return

    def get_state(self):
        return
    def get_observations(self):
        return self.history_obs

    def set_obs(self,images,time,save=False):
        self.history_obs = []
        for i, image in enumerate(images):
        # 如果是 numpy 数组，先转为 PIL.Image
            if isinstance(image, np.ndarray):
                image_rgb = Image.fromarray(image)

            # 确保没有 alpha 通道
            if image_rgb.mode == 'RGBA':
                image_rgb = image_rgb.convert('RGB')

            self.history_obs.append(image_rgb)
            if save:
                # plot_dir = os.path.join(self.plot_dir, f"episode_{self.episode_id}","sample")
                os.makedirs(self.plot_dir, exist_ok=True)
                imageio.imwrite(f'{self.plot_dir}/{len(self.history_obs)}.png', image_rgb)
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

        plot_world = np.concatenate([self.world_actions[:, :1], self.world_actions[:, 2:]], axis=1)
        world_img_np = plot_obs(time, plot_world, "follow the human", cur_image,human_position)
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


    def step(self,t, originla_quat = True):
        return

        
    def eval_bc_raw(self):

        assert self.instruction  is not None, "raw lang is None!!!!!!"
        set_seed(0)
        rand_crop_resize = False

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

                all_actions, outputs = self.policy.policy.evaluate(**batch, is_eval=True, tokenizer=self.policy.tokenizer)
                all_actions = all_actions.squeeze(0)  #
                all_actions = all_actions.to(dtype=torch.float32).cpu().numpy()
                all_actions = np.array([self.post_process(raw_action) for raw_action in all_actions])
                all_actions = all_actions - all_actions[0]
                # actions,raw_lang = self.get_info()
                # plot_actions(i,all_actions[0], actions, raw_lang, self.post_process, frames)
            self.local_actions = all_actions
            ####################################################################################################################################
            # clear previous actions
            while len(self.action_queue) > 0:
                self.action_queue.popleft()

            local_actions = self.local_actions.copy()


            return  local_actions
            ####################################################################################

    def visualize_trajectory(self, cv_image, all_actions):
            if cv_image is None or all_actions is None:
                return
            # now = time.time()
            # if not hasattr(self, "last_video_write_time"):
            #     self.last_video_write_time = 0.0
            # if now - self.last_video_write_time < 0.1:
            #     return
            # self.last_video_write_time = now
            img = cv_image.copy()
            h, w, _ = img.shape
            center_x, center_y = w // 2, h - 1
            scale = 50.0
            for i in range(len(all_actions) - 1):
                x1, y1 = all_actions[i, :2]
                x2, y2 = all_actions[i + 1, :2]
                p1 = (int(center_x + x1 * scale), int(center_y - y1 * scale))
                p2 = (int(center_x + x2 * scale), int(center_y - y2 * scale))
                cv2.line(img, p1, p2, (0, 255, 0), 2)
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            text = f"Linear: {self.linear_x:.2f} m/s | Angular: {self.w_angular:.2f} rad/s"
            cv2.rectangle(img, (10, 10), (450, 50), (0, 0, 0), -1)
            cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
