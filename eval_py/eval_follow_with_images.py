from qwen2_vla.model_load_utils import load_model_for_eval
from torchvision import transforms
import pickle
import time
from data_utils.utils import set_seed

from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *
import h5py
import cv2
from evaluate.visualize_action import plot_actions, plot_obs


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
    def datastruct_droid2qwen2vla(self, raw_lang,len_image=8):

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

        messages = self.datastruct_droid2qwen2vla(raw_lang)
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


def eval_bc(i,policy, target, deploy_env, policy_config, raw_lang=None, query_frequency=16, target_actions = None):

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
        
        all_actions = all_actions - all_actions[0]
        result = plot_obs(i, all_actions,raw_lang, obs['top'][-1], human_position = target, target_actions=target_actions)
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

        # 判断是否需要 decode（HDF5 字符串有时是 bytes）
        if isinstance(frame_paths_raw[0], bytes):
            frame_paths = [s.decode('utf-8') for s in frame_paths_raw]
        else:
            frame_paths = frame_paths_raw.tolist()

        follow_paths_group = f['follow_paths']
        for ep_key in follow_paths_group.keys():
            ep_group = follow_paths_group[ep_key]

            try:
                obs_idx = ep_group['obs_idx'][()]
                rel_path = ep_group['rel_path'][()]
                images = ep_group['observations/images'][()].decode('utf-8')
                actions = ep_group['action'][()]
                # 保存到结果字典中
                results[ep_key] = {
                    'obs_idx': obs_idx,
                    'rel_path': rel_path[-1],
                    'images': images,
                    'actions': actions
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
from pathlib import Path
def collect_n_frame_paths(image_folder, n_frames):
    image_folder = Path(image_folder)
    image_paths = list(image_folder.glob("*.jpg"))
    image_paths = sorted(
        image_paths,
        key=lambda x: float(x.stem)  # .stem 去掉后缀，只保留 '0.0'
    )
    n = len(image_paths)
    grouped_paths = []

    for i in range(len(image_paths)):
        if i + 1 < n_frames:
            pad = [image_paths[0]] * (n_frames - i - 1)
            frames = pad + image_paths[:i + 1]
        else:
            frames = image_paths[i + 1 - n_frames:i + 1]

        grouped_paths.append([str(p) for p in frames])  # 转换为字符串

    # 获取输出目录：与 image_folder 同级，名为 processed_frames
    output_dir = image_folder.parent / "predicted_mirror"
    output_dir.mkdir(exist_ok=True)

    return grouped_paths, str(output_dir)

if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 30
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        # "model_path": "OUTPUT/qwen2_follow_real/checkpoint-4000",
        "model_path": "/mnt/pfs/3zpd5q/code/train/DexVLA/OUTPUT/qwen2_follow_real_finetune_on_sample_mirror/checkpoint-4000",
        # "model_path": "OUTPUT/qwen2_new_follow_sample/checkpoint-20000",
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

    #### 3. Load DexVLA####################
    policy = qwen2_vla_policy(policy_config)
    
    #######################################
    import os
    for k in range(35):
        folder_path = f'test/sample{k}'

        grouped_paths, output_root =  collect_n_frame_paths(folder_path,8)
        os.makedirs(output_root, exist_ok=True)
        for i, frames in enumerate(grouped_paths):

            raw_lang = "follow the human"
            raw_lang = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."
            
            compressed = False
            images = []
            rank = 0
            for img_path in frames:
                img = cv2.imread(img_path)
                if compressed:
                    img = cv2.imdecode(img, 1)
                img = cv2.resize(img,  eval("(320,240)"))
                images.append(img)

            # target = [ep_data["rel_path"][0], ep_data["rel_path"][1], ep_data["rel_path"][2]]
            agilex_bot.set_obs(images, np.array([0,0,0]))
            # agilex_bot.set_info(actions, language_raw)
            result_image = eval_bc(i,policy, None,  agilex_bot, policy_config, raw_lang=raw_lang, query_frequency=query_frequency, target_actions=None)


            out_path = os.path.join(output_root, Path(frames[-1]).name)
            cv2.imwrite(out_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        # break
