from qwen2_vla.model_load_utils import load_model_for_eval
from torchvision import transforms
import pickle
import time
from data_utils.utils import set_seed
import argparse
from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *
import h5py
import cv2
from evaluate.visualize_action import plot_actions, plot_obs
from habitat_for_sim.utils.goat import read_yaml
from evaluate.visualize_action import visualize_trajectory
from label_with_vlm.mimo_action import MimoActionClient
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

import cv2
import numpy as np


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
    def datastruct_droid2qwen2vla(self, raw_lang,len_image=10):

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

        return messages
    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        
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
        messages = self.datastruct_droid2qwen2vla(raw_lang,len_image=len(image_list))
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


def eval_bc(i,policy, target, surpervised_action, deploy_env, policy_config, raw_lang=None,description=None , query_frequency=16):

    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)
    rand_crop_resize = False

    policy.policy.eval()

    ## 4. load data stats(min,max,mean....) and define post_process####################################
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'scale_dp_policy' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    #############################################################################################################



    image_list = []  # for visualization

    with torch.inference_mode():

        obs,states = deploy_env.get_obs()

        ### 5. Realize the function of get_obs###################
        traj_rgb_np, robot_state = process_obs(obs, states, stats)
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
        batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
        if policy_config['tinyvla']:
            all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=policy.tokenizer)
        else:
            # from inspect import signature
            # print(signature(policy.policy.generate))
            all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)

            all_actions = all_actions.squeeze(0)  #
            all_actions = all_actions.to(dtype=torch.float32).cpu().numpy()
            all_actions = np.array([post_process(raw_action) for raw_action in all_actions])
        
        smooth_action = smooth_yaw(all_actions, 3)
        # result = plot_obs(i, predicted_actions=all_actions ,raw_lang=description,obs = obs['top'][-1], human_position=target, target_actions=surpervised_action)
        result = visualize_trajectory(cv_image=obs['top'][-1], all_actions= smooth_action, instruction=description, surpervised_action = surpervised_action, subreason = outputs)
        return result



class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self):
        self.images = None
        self.states = None
        self.actions = None
        self.origin_lang = None
    def step(self, action):
        print("Execute action successfully!!!")

    def reset(self):
        print("Reset to home position.")

    def set_obs(self,images,qpos):
        self.images = images
        self.states = qpos
    def get_obs(self):
        obs = {
            'top': self.images,
        }
        states = self.states
        return obs, states
    
    def set_info(self,actions,origin_lang):
        self.actions = actions
        self.origin_lang = origin_lang
    def get_info(self):
        return self.actions, self.origin_lang
def extract_obs_and_paths(h5_file_path):
    results = {}
    with h5py.File(h5_file_path, 'r') as f:
        frame_paths_ds = f['frame_paths']
        frame_paths_raw = frame_paths_ds[:]  # 获取原始数据
        if len(frame_paths_raw) ==0:
            return None, None
        # 处理 frame_paths 解码
        if isinstance(frame_paths_raw[0], bytes):
            frame_paths = [s.decode('utf-8') for s in frame_paths_raw]
        else:
            frame_paths = frame_paths_raw.tolist()

        follow_paths_group = f['follow_paths']

        # ⭐ 对 ep_key 自然排序
        ep_keys = sorted(follow_paths_group.keys(), key=natural_key)

        for ep_key in ep_keys:
            ep_group = follow_paths_group[ep_key]

            try:
                obs_idx = ep_group['obs_idx'][()]
                rel_path = ep_group['action'][()]
                images = ep_group['observations/images'][()].decode('utf-8')

                raw_lang = ep_group['language_raw'][()]
                if isinstance(raw_lang, np.ndarray):
                    raw_lang = raw_lang.tolist()
                    raw_lang = random.choice(raw_lang)
                    # raw_lang = raw_lang[1]

                raw_lang = raw_lang.decode('utf-8')

                # 保存
                results[ep_key] = {
                    'obs_idx': obs_idx,
                    # 'target': rel_path[-1],
                    'action': rel_path,
                    'images': images,
                    'raw_lang': raw_lang
                }

            except KeyError as e:
                print(f"[WARN] Missing key in episode {ep_key}: {e}")

    return frame_paths, results
def get_history_frames(frame_paths, idx, n_frames):
    """
    返回包含 idx 在内，向前回溯的 n_frames 张图路径，
    不足则用 frame_paths[0] 补齐前面。
    """
    start = max(0, idx - n_frames + 1)
    selected = frame_paths[start:idx + 1]

    # 补齐前面不足的
    if len(selected) < n_frames:
        pad = [frame_paths[0]] * (n_frames - len(selected))
        selected = pad + selected

    return selected
import re

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True,
                        help='Path to the YAML config file')
    args = parser.parse_args()

    cfg = read_yaml(args.yaml_file_path)
    output_root = cfg.img_output_dir


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 30
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        # "model_path": "OUTPUT/qwen2_follow_real_finetune/checkpoint-10000",
        "model_path": cfg.model_path,
        #############################################################################
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
    }


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #### 2. Initialize robot env(Required)##########
    # Real aloha robot env
    # sys.path.insert(0, "/path/to/Dev-Code/mirocs")
    # from run.agilex_robot_env import AgilexRobot
    # agilex_bot = AgilexRobot()

    # fake env for debug
    agilex_bot = FakeRobotEnv()
    ######################################
    agilex_bot.reset()
    # mimo_action = MimoActionClient()
    #### 3. Load DexVLA####################
    policy = qwen2_vla_policy(policy_config)
    #######################################
    import os

    folder_path = cfg.proc_data_path


    folder_path = cfg.proc_data_path
    # test_files = sorted(
    #     [os.path.join(folder_path, f)
    #     for f in os.listdir(folder_path)
    #     if os.path.isfile(os.path.join(folder_path, f))],
    #     key=lambda x: natural_key(os.path.basename(x))
    # )
    test_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    os.makedirs(output_root, exist_ok=True)
    for i, file_path in enumerate(test_files):
        # if i <1000:
        #     continue
        # def print_hdf5_structure(file_path):
        #     def print_attrs(name, obj):
        #         print(name)
        #     with h5py.File(file_path, 'r') as f:
        #         f.visititems(print_attrs)
        
        # 示例使用
        # print_hdf5_structure(file_path)

        frames_paths, data = extract_obs_and_paths(file_path)
        if frames_paths is None:
            continue
        frames_paths = frames_paths[1:]
        file_output_dir = os.path.join(output_root, f"file_{i:03d}")
        os.makedirs(file_output_dir, exist_ok=True)
        n_frames = cfg.image_lens
        for ep_id, (ep_key, ep_data) in enumerate(data.items()):
            # if ep_data["obs_idx"] ==0:
            #     continue
            instruction = ep_data["raw_lang"]
            description = instruction
            # raw_lang = f"Your task is: {instruction}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."
            raw_lang = f"Your task is: {instruction}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, output the pixel  coordinate of your target on the last image."
            
            frames = get_history_frames(frames_paths, ep_data["obs_idx"]-1, n_frames)
            compressed = False
            images = []
            rank = 0
            for img_path in frames:
                # img_path = img_path.replace("frames/", f"frames_{rank}/")
                img = cv2.imread(img_path)
                if compressed:
                    img = cv2.imdecode(img, 1)
                img = cv2.resize(img,  eval("(320,240)"))
                images.append(img)

            # target = [ep_data["target"][0], ep_data["target"][1], ep_data["target"][2]]
            target = None
            agilex_bot.set_obs(images, np.array([0,0,0]))
            # agilex_bot.set_info(actions, language_raw)
            result_image = eval_bc(i,policy, target = None, surpervised_action= ep_data['action'], deploy_env= agilex_bot,policy_config= policy_config, raw_lang=raw_lang,description=description ,query_frequency=query_frequency)
            
            # coords, answer = mimo_action.get_action(instruction, frames[-1], frames[:-1])
            # img_np = np.array(result_image)
            
            # H, W = img_np.shape[:2]
            # if coords:
            #     h, w = coords[:2]
            #     overlay = img_np.copy()
            #     cv2.circle(overlay, (int(w*3/2), int(h*3/2)), 10, (0, 0, 255), -1)
            # else:
            #     cv2.putText(overlay,
            #         answer,
            #         (10, H - 10),                 # 左下角，留一点边距
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.8,                          # 字体大小
            #         (255, 0, 0),                  # 颜色 (B, G, R)
            #         2,                            # 线宽
            #         cv2.LINE_AA
            #     )
            out_path = os.path.join(file_output_dir, f"ep_{ep_id:04d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(out_path, overlay)
        # break

