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

        self.config = AutoConfig.from_pretrained(
            '/'.join(model_path.split('/')[:-1]), 
            trust_remote_code=True,
            local_files_only=True
            )
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

        messages = self.datastruct_droid2qwen2vla(raw_lang,len_image=10)
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
    image_paths = list(image_folder.glob("*.png"))
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
    output_dir = image_folder.parent / "predicted"
    output_dir.mkdir(exist_ok=True)

    return grouped_paths, str(output_dir)

if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 30
    policy_config = {
        "model_path": "/mnt/pfs/3zpd5q/code/zf/DexVLA/OUTPUT/qwen2_dexvln_door_with_rota_move/checkpoint-15000",
        "model_base": None,
        "enable_lora": False,
        "action_head": action_head,
        "tinyvla": False,
    }
    data_set = "/mnt/pfs/3zpd5q/code/zf/train_data/splite_data"
    save_path = "/mnt/pfs/3zpd5q/code/zf/DexVLA/test/test_door_rotate_move"
    os.makedirs(save_path, exist_ok=True)
    n_episodes = 50
    n_frames = 10

    all_episodes = list(Path(data_set).glob("episode_*.hdf5"))
    sampled_episodes = random.sample(all_episodes, min(n_episodes, len(all_episodes)))
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #### 2. Initialize robot env(Required)##########
    agilex_bot = FakeRobotEnv()
    agilex_bot.reset()

    #### 3. Load DexVLA####################
    policy = qwen2_vla_policy(policy_config)
    #######################################
    for ep_file in sampled_episodes:
        with h5py.File(ep_file, 'r') as f:
            history_images = f['observations/history_images'][:]
            qpos = f['observations/qpos'][:]
            all_action = f['/action'][()]

            # decode if bytes
            if isinstance(history_images[0], bytes):
                history_images = [s.decode('utf-8') for s in history_images]

            # 随机选择一个 step
            step_idx = random.randint(0, len(history_images)//2)
            selected_frames = get_history_frames(history_images, step_idx, n_frames)
            # 读取图片
            images = []
            for img_path in selected_frames:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (320, 240))
                images.append(img)

            # 构造语言指令
            raw_lang = f['language_raw'][()].decode('utf-8')
            raw_lang = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."
            # target_action
            action = all_action[step_idx]
            # action = action[step_idx:]

            # x0, y0, yaw0 = base_pose[:3]
            # transformed_actions = []
            # for a in action:
            #     dx = a[0] - x0
            #     dy = a[1] - y0
            #     # 将全局坐标转换到局部坐标
            #     x_new =  np.cos(yaw0) * dx + np.sin(yaw0) * dy
            #     y_new = -np.sin(yaw0) * dx + np.cos(yaw0) * dy
            #     yaw_new = a[2] - yaw0
            #     transformed_actions.append([x_new, y_new, yaw_new])
            # action = np.array(transformed_actions, dtype=np.float32)
            # action[0, :3] = 0.0
            
            # max_len = 30
            # if len(action) < max_len:
            #     last_action = action[-1]
            #     repeat_count = max_len - len(action)
            #     action = np.concatenate([action, np.tile(last_action, (repeat_count, 1))], axis=0)
            # else:
            #     action = action[:max_len]
            # 机器人状态（假设为 0,0,0，也可以用 qpos[step_idx]）
            agilex_bot.set_obs(images, np.array([0,0,0]))

            # 模型预测
            result_image = eval_bc(
                step_idx, policy, None, agilex_bot, policy_config, raw_lang=raw_lang,
                query_frequency=query_frequency, target_actions=action
            )

            # 保存结果到指定目录 save_path
            out_path = Path(save_path) / Path(selected_frames[-1]).name
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        # break

