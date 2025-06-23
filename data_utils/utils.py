import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import multiprocessing as mp
import IPython
import copy
e = IPython.embed
from aloha_scripts.utils import *

def flatten_list(l):
    return [item for sublist in l for item in sublist]
import gc
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class, robot=None, rank0_print=print, llava_pythia_process=None, data_args=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.llava_pythia_process = llava_pythia_process
        self.data_args = data_args
        self.robot = robot
        self.rank0_print = rank0_print
        if 'diffusion' in self.policy_class.lower() or 'scale_dp' in self.policy_class.lower():
            self.augment_images = True
        else:
            self.augment_images = False

        self.augment_images = False #wzj

        self.transformations = None
        self.rank0_print(f"########################Current Image Size is [{self.data_args.image_size_stable}]###################################")
        self.rank0_print(f"{RED}policy class: {self.policy_class}; augument: {self.augment_images}{RESET}")
        # a=self.__getitem__(0) # initialize self.is_sim and self.transformations
        # self.rank0_print('Initializing transformations')
        # original_size = eval(self.data_args.image_size_stable)  # e.g., (320, 240)
        # ratio = 0.95
        # self.transformations = [
        #     transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
        #     transforms.Resize(original_size, antialias=True),
        #     transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
        #     transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
        # ]



        if len(self.camera_names) > 2:
            # self.rank0_print("%"*40)
            self.rank0_print(f"The robot is {RED} {self.robot} {RESET} | The camera views: {RED} {self.camera_names} {RESET} | The history length: {RED} {self.data_args.history_images_length} {RESET}")
        self.is_sim = False

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def _load_from_h5(self, dataset_path, start_ts):
        with h5py.File(dataset_path, 'r') as root:
            try: # some legacy data does not have this attribute
                is_sim = root.attrs['sim']
            except:
                is_sim = False
            compressed = root.attrs.get('compress', False)
            if 'truncate' in dataset_path:
                compressed = False
            try:
                raw_lang = root['language_raw'][0].decode('utf-8')
            except Exception as e:
                # self.rank0_print(e)
                self.rank0_print(f"Read {dataset_path} happens {YELLOW}{e}{RESET}")
                exit(0)
            reasoning = " "
            if self.data_args.use_reasoning:
                if 'substep_reasonings' in root.keys():
                    reasoning = root['substep_reasonings'][start_ts].decode('utf-8')
                else:
                    try:
                        reasoning = root['reasoning'][0].decode('utf-8')
                    except Exception as e:
                        self.rank0_print(f"Read reasoning from {dataset_path} happens {YELLOW}{e}{RESET}")
                        exit(0)
            action = root['/action'][()]
            original_action_shape = action.shape
            episode_len = original_action_shape[0]

            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                image_dict[cam_name] = cv2.resize(image_dict[cam_name], eval(self.data_args.image_size_stable))

            if compressed:
                print(f"{RED} It's compressed in {dataset_path} {RESET}")
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
        return original_action_shape, action, action_len, image_dict, qpos, qvel, raw_lang, reasoning

    def _load_from_h5_nav(self, dataset_path, start_ts):
        start_ts = 0
        with h5py.File(dataset_path, 'r') as root:
            try: # some legacy data does not have this attribute
                is_sim = root.attrs['sim']
            except:
                is_sim = False
            compressed = root.attrs.get('compress', False)
            if 'truncate' in dataset_path:
                compressed = False
            try:
                raw_lang = root['language_raw'][()].decode('utf-8')
                #wzj
                raw_lang = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."
                # instruction = root['instruction'][()].decode('utf-8')
            except Exception as e:
                # self.rank0_print(e)
                self.rank0_print(f"Read {dataset_path} happens {YELLOW}{e}{RESET}")
                exit(0)
            reasoning = " "
            if self.data_args.use_reasoning:
                if 'substep_reasonings' in root.keys():
                    reasoning = root['substep_reasonings'][start_ts].decode('utf-8')
                else:
                    try:
                        reasoning = root['reasoning'][0].decode('utf-8')
                    except Exception as e:
                        self.rank0_print(f"Read reasoning from {dataset_path} happens {YELLOW}{e}{RESET}")
                        exit(0)
            # action = root['/action'][()][:, :2] #wzj xy
            action = root['/action'][()]
            original_action_shape = action.shape
            episode_len = original_action_shape[0]

            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            qvel = root['/observations/qpos'][start_ts]
            image_dict = dict()
            # video_dict = dict()
            n_frames = self.data_args.history_images_length

            cam_name = self.camera_names[0]

            image_seq = root[f'/observations/images/{cam_name}'][()]
            history_image_seq = root[f'/observations/history_images'][()]

            frames = []


            assert n_frames<=len(history_image_seq)

            # if n_frames >= len(history_image_seq):  # 临时
            #     history_image_seq[0] = history_image_seq[1]
            #     if n_frames > len(history_image_seq):
            #         history_image_seq = [history_image_seq[0]] * (n_frames - len(history_image_seq)) + history_image_seq

            for path_bytes in history_image_seq[-n_frames:]:
                img_path = path_bytes.decode('utf-8')
                img = cv2.imread(img_path)
                if compressed:
                    img = cv2.imdecode(img, 1)
                img = cv2.resize(img,  eval(self.data_args.image_size_stable))
                frames.append(img)

            img_path = image_seq.decode('utf-8')
            img = cv2.imread(img_path)
            img = cv2.resize(img,  eval(self.data_args.image_size_stable))
            frames.append(img)


            # 存储单帧图像（最后一帧）
            image_dict[cam_name] = frames



            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
        return original_action_shape, action, action_len, image_dict, frames, qpos, qvel, raw_lang, reasoning




    def _load_worker_wrapper(queue, dataset_path, start_ts, self_ref):
        try:
            result = self_ref._load_from_h5(dataset_path, start_ts)
            queue.put((True, result))
        except Exception as e:
            queue.put((False, e))

    def safe_load_with_retries(self, dataset_path, start_ts, max_retries=3, timeout=10):
        ctx = mp.get_context('spawn')  # 避免 fork 导致 HDF5 共享问题
        for attempt in range(max_retries):
            queue = ctx.Queue()
            p = ctx.Process(target=self._load_worker_wrapper, args=(queue, dataset_path, start_ts, self))
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                self.rank0_print(f"[Retry {attempt+1}] Timeout when reading {dataset_path}")
                continue
            if not queue.empty():
                success, payload = queue.get()
                if success:
                    return payload
                else:
                    self.rank0_print(f"[Retry {attempt+1}] Exception while reading {dataset_path}: {repr(payload)}")

            else:
                self.rank0_print(f"[Retry {attempt+1}] No result returned from {dataset_path}")
        raise RuntimeError(f"Failed to load {dataset_path} after {max_retries} retries.")

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        fallback_offsets = [0, 1, -1]

        for offset in fallback_offsets:
            new_id = episode_id + offset
            if not (0 <= new_id < len(self.dataset_path_list)):
                continue
            dataset_path = self.dataset_path_list[new_id]

            try:
                # (
                #     original_action_shape,
                #     action,
                #     action_len,
                #     image_dict,
                #     qpos,
                #     qvel,
                #     raw_lang,
                #     reasoning
                # ) = self._load_from_h5(dataset_path, start_ts)
                (
                    original_action_shape,
                    action,
                    action_len,
                    image_dict,
                    video,
                    qpos,
                    qvel,
                    raw_lang,
                    reasoning
                ) = self._load_from_h5_nav(dataset_path, start_ts)
                if raw_lang is None or action is None or image_dict is None:
                    raise ValueError(f"Incomplete sample from {dataset_path}")
                break
            except Exception as e:
                self.rank0_print(f"[Rank {getattr(self, 'rank', 'N/A')}] Fallback {offset} failed: {dataset_path} | {e}")
                self.rank0_print(f"[Rank {getattr(self, 'rank', 'N/A')}] Tried files: {[self.dataset_path_list[episode_id + o] for o in fallback_offsets if 0 <= episode_id + o < len(self.dataset_path_list)]}")

        else:
            raise RuntimeError(
                f"[Rank {getattr(self, 'rank', 'N/A')}] All fallback loading failed for index {index} "
                f"(episode_id={episode_id}, tried offsets={fallback_offsets})"
            )



        # self.is_sim = is_sim
        padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
        if self.data_args.delta_control:
            padded_action[:action_len - 1] = action[1:] - action[:-1]
        else:
            padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len)
        is_pad[action_len:] = 1

        padded_action = padded_action[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            if cam_name in image_dict:
                for img in image_dict[cam_name]:
                    all_cam_images.append(img) #wzj
        all_cam_images = np.stack(all_cam_images, axis=0)
        video = np.stack(video, axis=0)
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        video_data = torch.from_numpy(video)

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # image_data = torch.einsum('k h w c -> k c h w', image_data)
        # video_data = torch.einsum('k h w c -> k c h w', video_data)
        image_data = image_data.permute(0, 3, 1, 2)  # [T, H, W, C] → [T, C, H, W]
        video_data = video_data.permute(0, 3, 1, 2)  # [T, H, W, C] → [T, C, H, W]


        # # augmentation

        # if not hasattr(self, "transformations") or self.transformations is None:
        #     self.rank0_print('Initializing transformations')
        #     original_size = image_data.shape[-2:]
        #     ratio = 0.95
        #     self.transformations = [
        #         transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
        #         transforms.Resize(original_size, antialias=True),
        #         transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
        #         transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
        #     ]


        # if self.augment_images:
        #     for transform in self.transformations:
        #         image_data = transform(image_data)

        if not hasattr(self, "transformations") or self.transformations is None:
            self.rank0_print('Initializing transformations')
            original_size = video_data.shape[-2:]
            ratio = 0.95
            self.transformations = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize(original_size, antialias=True),
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            ]


        if self.augment_images:
            for transform in self.transformations:
                video_data = transform(video_data)


        norm_stats = self.norm_stats

        action_data = ((action_data - norm_stats["action_min"]) / (norm_stats["action_max"] - norm_stats["action_min"])) * 2 - 1

        qpos_data = (qpos_data - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]

        sample = {
            'image': image_data,
            'video': video_data,
            'state': qpos_data,
            'action': action_data,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': reasoning
        }
        assert raw_lang is not None, ""
        if index == 0:
            self.rank0_print(reasoning)
        del image_data
        del video_data
        del qpos_data
        del action_data
        del is_pad
        del raw_lang
        del reasoning
        gc.collect()
        torch.cuda.empty_cache()

        return self.llava_pythia_process.forward_process(sample, use_reasoning=self.data_args.use_reasoning)


def get_norm_stats(dataset_path_list, rank0_print=print):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                # qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
                # #wzj xy
                # action = action[:, :2]
                # qpos = qpos[:, :2]
        except Exception as e:
            rank0_print(f'Error loading {dataset_path} in get_norm_stats')
            rank0_print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

# calculating the norm stats corresponding to each kind of task (e.g. folding shirt, clean table....)
def get_norm_stats_by_tasks(dataset_path_list):

    data_tasks_dict = dict(
        fold_shirt=[],
        clean_table=[],
        others=[],
    )
    for dataset_path in dataset_path_list:
        if 'fold' in dataset_path or 'shirt' in dataset_path:
            key = 'fold_shirt'
        elif 'clean_table' in dataset_path and 'pick' not in dataset_path:
            key = 'clean_table'
        else:
            key = 'others'
        data_tasks_dict[key].append(dataset_path)

    norm_stats_tasks = {k : None for k in data_tasks_dict.keys()}

    for k,v in data_tasks_dict.items():
        if len(v) > 0:
            norm_stats_tasks[k], _ = get_norm_stats(v)

    return norm_stats_tasks


def find_all_hdf5(dataset_dir, skip_mirrored_data, rank0_print=print):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        if 'pointcloud' in root: continue
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    if len(hdf5_files) == 0:
        rank0_print(f"{RED} Found 0 hdf5 datasets found in {dataset_dir} {RESET}")
        exit(0)
    rank0_print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def filter_valid_hdf5(dataset_path_list, rank0_print=print):
    valid_paths = []
    for path in dataset_path_list:
        try:
            with h5py.File(path, 'r') as f:
                # 简单读取关键字段进行验证
                if '/action' not in f:
                    raise ValueError("Missing /action key")
                _ = f['/action'][()]  # 检查能否读取
        except Exception as e:
            rank0_print(f"[WARN] Skipping broken hdf5 file: {path} | {type(e).__name__}: {e}")
            continue
        valid_paths.append(path)
    return valid_paths


def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, config, rank0_print=print, skip_mirrored_data=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99, llava_pythia_process=None):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    # find all data
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data, rank0_print=rank0_print) for dataset_dir in dataset_dir_l]
    for d,dpl in zip(dataset_dir_l, dataset_path_list_list):
        if len(dpl) == 0:
            rank0_print("#2"*20)
            rank0_print(d)

    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    dataset_path_list = filter_valid_hdf5(dataset_path_list, rank0_print)
    rank0_print(f"{RED}Valid HDF5 files: {len(dataset_path_list)} (filtered from total {sum(len(x) for x in dataset_path_list_list)}){RESET}")



    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]

    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    rank0_print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    _, all_episode_len = get_norm_stats(dataset_path_list)
    rank0_print(f"{RED}All images: {sum(all_episode_len)}, Trajectories: {len(all_episode_len)} {RESET}")
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]

    
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]

    # calculate norm stats across all episodes
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data, rank0_print=rank0_print) for stats_dir in stats_dir_l]))

    # calculate norm stats corresponding to each kind of task
    rank0_print(f'Norm stats from: {[each.split("/")[-1] for each in stats_dir_l]}')
    rank0_print(f'train_episode_len_l: {train_episode_len_l}') #wzjprint


    robot = 'aloha' if config['action_head_args'].action_dim == 14 or ('aloha' in config['training_args'].output_dir) else 'franka'
    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class, robot=robot, llava_pythia_process=llava_pythia_process, data_args=config['data_args'])
    # val_dataset is unused
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class, robot=robot, llava_pythia_process=llava_pythia_process, data_args=config['data_args'])

    sampler_params = {
        'train': {"batch_size": batch_size_train, 'episode_len_l': train_episode_len_l, 'sample_weights':sample_weights, 'episode_first': config['data_args'].episode_first},
        'eval': {"batch_size": batch_size_val, 'episode_len_l': val_episode_len_l, 'sample_weights': None, 'episode_first': config['data_args'].episode_first} # unused
    }
    return train_dataset, val_dataset, norm_stats, sampler_params


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0

    return np.array([linear_vel, angular_vel])

### env utils

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
