from qwen2_vla.model_load_utils import load_model_for_eval
from torchvision import transforms
import pickle
import time
from data_utils.utils import set_seed

from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *
import h5py
import cv2
from visualize_action import plot_actions


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
    def datastruct_droid2qwen2vla(self, raw_lang):
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": None,
        #             },
        #             {"type": "text", "text": f""},
        #         ],
        #     },
        #     # {"role": "assistant", "content": f''},
        # ]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {"type": "text", "text": f""},
                ],
            },
            # {"role": "assistant", "content": f''},
        ]
        messages[0]['content'][-1]['text'] = raw_lang

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


def eval_bc(i,policy, deploy_env, policy_config, raw_lang=None, query_frequency=16):

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

    from collections import deque
    action_queue = deque(maxlen=query_frequency)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks

    for rollout_id in range(1000):

        rollout_id += 0

        image_list = []  # for visualization

        with torch.inference_mode():
            time0 = time.time()
            for t in range(max_timesteps):

                obs,states = deploy_env.get_obs()

                ### 5. Realize the function of get_obs###################
                traj_rgb_np, robot_state = process_obs(obs, states, stats)
                #########################################################
                image_list.append(traj_rgb_np)
                robot_state = torch.from_numpy(robot_state).float().cuda()
                if t % query_frequency == 0:
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

                        action_queue.extend(
                                torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:30])
                        

                        actions,raw_lang = deploy_env.get_info()
                        plot_actions(i,all_actions[0], actions, raw_lang, post_process, frames)
                        break
                    ####################################################################################################################################
                    # clear previous actions
                    while len(action_queue) > 0:
                        action_queue.popleft()

                    action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency])

                raw_action = action_queue.popleft()
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                ### 8. post process actions##########################################################
                action = post_process(raw_action)
                #####################################################################################
                print(f"after post_process action size: {action.shape}")
                print(f'step {t}, pred action: {outputs}{action}')
                if len(action.shape) == 2:
                    action = action[0]
                ##### Execute ######################################################################
                action_info = deploy_env.step(action.tolist())
                ####################################################################################
            break

class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self):
        self.images = None
        self.states = None
        self.actions = None
        self.raw_lang = None
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
    
    def set_info(self,actions,raw_lang):
        self.actions = actions
        self.raw_lang = raw_lang
    def get_info(self):
        return self.actions, self.raw_lang

if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 16
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        "model_path": "OUTPUT/qwen2_dexvln_debug/checkpoint-5000",
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

    folder_path = 'data/test2'
    test_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for i, file_path in enumerate(test_files):

        with h5py.File(file_path, 'r') as f:
            instruction = f['instruction']
            observsation = f['/observations/images/cam_high'][()]
            history_images = f['/observations/history_images'][()]   # 假设 'annotations' 存在\
            language_raw = f['language_raw'][()].decode('utf-8')
            actions = f['/action'][()]# 假设 'type' 存在
            qposes = f['/observations/qpos'][()]

        raw_lang =language_raw
        raw_lang = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."
        frames = []


        n_frames = 5
        compressed = False
        for path_bytes in history_images[-n_frames:]:
            img_path = path_bytes.decode('utf-8')
            img = cv2.imread(img_path)
            if compressed:
                img = cv2.imdecode(img, 1)
            img = cv2.resize(img,  eval("(320,240)"))
            frames.append(img)

        img_path = observsation.decode('utf-8')
        img = cv2.imread(img_path)
        img = cv2.resize(img,  eval("(320,240)"))
        frames.append(img)

        agilex_bot.set_obs(frames, qposes[0])
        agilex_bot.set_info(actions, language_raw)
        eval_bc(i,policy, agilex_bot, policy_config, raw_lang=raw_lang, query_frequency=query_frequency)


