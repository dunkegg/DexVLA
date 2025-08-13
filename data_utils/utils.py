import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import zarr
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import multiprocessing as mp
import IPython
import copy
import random
from tqdm import tqdm
e = IPython.embed
from aloha_scripts.utils import *
import json

import multiprocessing
def _zarr_open_worker(path, queue):
    try:
        arr = zarr.open(path, mode='r')
        queue.put(arr)
    except Exception as e:
        queue.put(e)

def safe_zarr_open(path, timeout=5):
    ctx = multiprocessing.get_context("spawn")  # 更安全地避免多线程死锁
    queue = ctx.Queue()
    p = ctx.Process(target=_zarr_open_worker, args=(path, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Timeout opening Zarr file: {path}")

    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result
def flatten_list(l):
    return [item for sublist in l for item in sublist]

import matplotlib.pyplot as plt

def plot_actions_and_save_frames(raw_lang, action, frames, save_dir):
    """
    绘制动作轨迹和 yaw，并保存图像帧
    
    参数:
        raw_lang (str): 任务说明
        action (np.ndarray): 动作轨迹数组 [T, D]，至少有三个维度 (x, y, yaw)
        frames (list[np.ndarray]): 图像帧列表
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    if action.shape[1] < 3:
        raise ValueError("Action must have at least three dimensions for plotting (x, y, yaw).")

    # === 1. 绘制轨迹和 yaw ===
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # XY 轨迹
    axs[0].plot(action[:, 0], action[:, 1], marker='o', markersize=3, linewidth=1)
    axs[0].set_title("XY Trajectory", fontsize=10)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].grid(True)
    axs[0].axis('equal')

    # yaw 曲线
    timesteps = np.arange(len(action))
    axs[1].plot(timesteps, action[:, 2], marker='o', markersize=3, linewidth=1, color='orange')
    axs[1].set_title("Yaw over Time", fontsize=10)
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Yaw (rad)")
    axs[1].grid(True)

    fig.suptitle(raw_lang, fontsize=12, wrap=True)
    traj_path = os.path.join(save_dir, "trajectory_yaw.png")
    plt.tight_layout()
    plt.savefig(traj_path, dpi=300)
    plt.close()
    print(f"✅ 动作轨迹 + Yaw 已保存: {traj_path}")

    # === 2. 保存图像帧 ===
    for i, frame in enumerate(frames):
        frame_path = os.path.join(save_dir, f"frame_{i:03d}.png")
        cv2.imwrite(frame_path, frame)
    print(f"✅ 共保存 {len(frames)} 张帧图像到: {save_dir}")
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

        self.same_type_count = 0
        self.train_type = True

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        # start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        start_ts = 0
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def _load_from_nav(self, dataset_path, start_ts=0):
        is_zarr = dataset_path.endswith(".zarr")
        raw_lang = ""
        reasoning = " "

        if is_zarr:
            # root = zarr.open(dataset_path, 'r')
            root = safe_zarr_open(dataset_path)
            get_attr = lambda k, default=None: root.attrs.get(k, default)
            get_item = lambda k: root[k][()]
            get_at = lambda k, i: root[k][i]
        else:
            with h5py.File(dataset_path, 'r') as root:
                return self._load_from_h5_internal(root, dataset_path, start_ts)


    def _load_from_h5_internal(self,root, dataset_path, start_ts):
        start_ts = 0

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
            old_raw_lang = raw_lang
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

        obs_img = root[f'/observations/images/{cam_name}'][()]
        history_image_seq = root[f'/observations/history_images'][()]

        frames = []


        assert n_frames<=len(history_image_seq)

        mirror = root.attrs.get("tag", "") == "mirror"

        # batch_size = 2
        # # if self.same_type_count == batch_size:
        # #     self.train_type = random.choice([True, False])
        # #     self.same_type_count=1
        # # self.same_type_count += 1
        
        # self.train_type = random.choice([True, False])
        
        # if self.train_type:
        #     n_frames = 4
        #     history_image_seq = history_image_seq[0:5]
        # else:
        #     n_frames = 3


        rank = int(os.environ.get("RANK", 0))
        for path_bytes in history_image_seq[-n_frames:]:
            img_path = path_bytes.decode('utf-8')
            # img_path = img_path.replace("code/", f"code/train/")
            img = cv2.imread(img_path)
            if mirror:
                img = cv2.flip(img, 1)
            if compressed:
                img = cv2.imdecode(img, 1)
            img = cv2.resize(img,  eval(self.data_args.image_size_stable))
            frames.append(img)

        # if not self.train_type:
        img_path = obs_img.decode('utf-8')
        # img_path = img_path.replace("code/", f"code/train/")
        img = cv2.imread(img_path)
        if mirror:
            img = cv2.flip(img, 1)
        if compressed:
            img = cv2.imdecode(img, 1)
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


        # plot_actions_and_save_frames(old_raw_lang, action, frames, 'check_hdf5')
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
                ) = self._load_from_nav(dataset_path, start_ts)
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


def get_norm_stats(dataset_path_list, rank0_print=print,  cache_path="norm_stats.json"):
    # 如果缓存文件存在，直接加载
    if os.path.exists(cache_path):
        rank0_print(f"[INFO] Loading cached norm stats from {cache_path}")
        with open(cache_path, "r") as f:
            stats = json.load(f)

        # 还原 numpy 格式
        keys_to_np = ["action_mean", "action_std", "action_min", "action_max", "qpos_mean", "qpos_std", "example_qpos"]
        for k in keys_to_np:
            stats[k] = np.array(stats[k])

        # 还原 episode lens
        ep_len_val = stats.get("episode_len_value", None)
        ep_len_count = stats.get("episode_len_count", 0)
        episode_lens = [ep_len_val] * ep_len_count if ep_len_val is not None else None

        return stats, episode_lens    
    
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in tqdm(dataset_path_list, desc="Get Norm Loading datasets"):
        try:
            if dataset_path.endswith(".hdf5") or dataset_path.endswith(".h5"):
                with h5py.File(dataset_path, 'r') as root:
                    qpos = root['/observations/qpos'][()]
                    action = root['/action'][()]
            elif dataset_path.endswith(".zarr"):
                # root = zarr.open(dataset_path, mode='r')
                root = safe_zarr_open(dataset_path)
                qpos = root['/observations/qpos'][:]
                action = root['/action'][:]
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path}")
        except Exception as e:
            rank0_print(f'Error loading {dataset_path} in get_norm_stats')
            rank0_print(e)
            # continue
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
    # stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
    #          "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
    #          "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
    #          "example_qpos": qpos}
    stats = {
        "action_mean": action_mean.numpy().tolist(),
        "action_std": action_std.numpy().tolist(),
        "action_min": (action_min - eps).numpy().tolist(),
        "action_max": (action_max + eps).numpy().tolist(),
        "qpos_mean": qpos_mean.numpy().tolist(),
        "qpos_std": qpos_std.numpy().tolist(),
        "example_qpos": qpos.tolist(),
    }
    # 简化 episode_lens 缓存
    if len(set(all_episode_len)) == 1:
        stats["episode_len_value"] = all_episode_len[0]
        stats["episode_len_count"] = len(all_episode_len)
    else:
        rank0_print("[WARN] episode_len 不一致，未压缩缓存")
        stats["episode_lens"] = all_episode_len  # fallback to full list

    # 写入 JSON 缓存
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2)
    rank0_print(f"[INFO] Saved norm stats to {cache_path}")
    stats["action_mean"] = action_mean.numpy()
    stats["action_std"] = action_std.numpy()
    stats["action_min"] = (action_min - eps).numpy()
    stats["action_max"] = (action_max + eps).numpy()
    stats["qpos_mean"] = qpos_mean.numpy()
    stats["qpos_std"] = qpos_std.numpy()
    stats["example_qpos"] = qpos

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


# def find_all_hdf5(dataset_dir, skip_mirrored_data, rank0_print=print):
#     hdf5_files = []
#     for root, dirs, files in os.walk(dataset_dir):
#         if 'pointcloud' in root: continue
#         for filename in fnmatch.filter(files, '*.hdf5'):
#             if 'features' in filename: continue
#             if skip_mirrored_data and 'mirror' in filename:
#                 continue
#             hdf5_files.append(os.path.join(root, filename))
#     if len(hdf5_files) == 0:
#         rank0_print(f"{RED} Found 0 hdf5 datasets found in {dataset_dir} {RESET}")
#         exit(0)
#     rank0_print(f'Found {len(hdf5_files)} hdf5 files')
#     return hdf5_files


def find_all_hdf5(dataset_dir, skip_mirrored_data, rank0_print=print):
    dataset_paths = []

    for root, dirs, files in tqdm(os.walk(dataset_dir), desc="Walking through dataset"):
        # print(f"Processing {root}, {len(files)} files")
        if 'pointcloud' in root:
            continue

        # HDF5 文件匹配
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename:
                continue
            if skip_mirrored_data and 'mirror_2' in filename:
                continue
            dataset_paths.append(os.path.join(root, filename))

        # Zarr 目录匹配
        for dirname in dirs:
            if not dirname.endswith(".zarr"):
                continue
            if skip_mirrored_data and 'mirror' in dirname:
                continue
            dataset_paths.append(os.path.join(root, dirname))

    if len(dataset_paths) == 0:
        rank0_print(f"{RED} Found 0 datasets in {dataset_dir} {RESET}")
        exit(0)

    rank0_print(f'Found {len(dataset_paths)} dataset files (hdf5/zarr)')
    return dataset_paths

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
    for path in tqdm(dataset_path_list, desc="Filter Loading datasets"):
        try:
            if path.endswith(".hdf5") or path.endswith(".h5"):
                with h5py.File(path, 'r') as f:
                    if '/action' not in f:
                        raise ValueError("Missing /action key")
                    _ = f['/action'][()]  # 检查能否读取

            elif path.endswith(".zarr"):
                # root = zarr.open(path, mode='r')
                root = safe_zarr_open(path)
                if '/action' not in root:
                    raise ValueError("Missing /action key")
                _ = root['/action'][()]  # 检查能否读取

            else:
                raise ValueError(f"Unsupported dataset format: {path}")

        except Exception as e:
            rank0_print(f"[WARN] Skipping broken file: {path} | {type(e).__name__}: {e}")
            continue
        valid_paths.append(path)
    return valid_paths


def load_data(dataset_dir_l, name_filter, camera_names,
              batch_size_train, batch_size_val,
              chunk_size, config,
              rank0_print=print, skip_mirrored_data=False,
              policy_class=None, stats_dir_l=None,
              sample_weights=None, train_ratio=0.99,
              llava_pythia_process=None):

    if isinstance(dataset_dir_l, str):
        dataset_dir_l = [dataset_dir_l]

    # ------------------------------------------------------------------
    # 1) 构建 “每个数据集” 对应的 h5 列表，并在本层完成所有过滤
    # ------------------------------------------------------------------
    dataset_path_list_list = []
    total_before=0
    for d in dataset_dir_l:
        rank = int(os.environ.get("RANK", 0))
        print(f"!!!!!!!!!!!!!!!!!!!!!!!   Rank: {rank}, data path {d}")
        # d = d + f"_{rank}"
        raw_paths = find_all_hdf5(d, skip_mirrored_data, rank0_print=rank0_print)
        if len(raw_paths) == 0:
            rank0_print("#2"*20); rank0_print(d)
        total_before += len(raw_paths)  # 累加原始文件数
        # ① name_filter
        filtered = [p for p in raw_paths if name_filter(p)]
        # ② valid h5
        # filtered = filter_valid_hdf5(filtered, rank0_print)
        dataset_path_list_list.append(filtered)

    # # 打印统计
    # total_before = sum(len(find_all_hdf5(d, skip_mirrored_data)) for d in dataset_dir_l)
    # total_after  = sum(len(sub) for sub in dataset_path_list_list)
    # rank0_print(f"{RED}Valid HDF5 files: {total_after} (filtered from total {total_before}){RESET}")
    # 直接使用已缓存的数据打印统计
    total_after = sum(len(sub) for sub in dataset_path_list_list)
    rank0_print(f"{RED}Valid HDF5 files: {total_after} (filtered from total {total_before}){RESET}")
    # ------------------------------------------------------------------
    # 2) 统计每个子数据集 episode 数，再做 train/val split
    # ------------------------------------------------------------------
    num_episodes_l      = [len(sub) for sub in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # 只在第 0 个数据集上随机打散，保持与原逻辑一致
    num_episodes_0 = num_episodes_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    cut = int(train_ratio * num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:cut]
    val_episode_ids_0   = shuffled_episode_ids_0[cut:]

    # 其余数据集全部划入 train（与原实现相同）
    train_episode_ids_l = [train_episode_ids_0] + [
        np.arange(n) + num_episodes_cumsum[i]
        for i, n in enumerate(num_episodes_l[1:])
    ]
    val_episode_ids_l   = [val_episode_ids_0]

    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids   = np.concatenate(val_episode_ids_l)

    rank0_print(
        f'\n\nData from: {dataset_dir_l}'
        f'\n- Train on {[len(x) for x in train_episode_ids_l]} episodes'
        f'\n- Test  on {[len(x) for x in val_episode_ids_l]} episodes\n'
    )

    # ------------------------------------------------------------------
    # 3) 后续统计和数据加载逻辑保持不变
    # ------------------------------------------------------------------
    dataset_path_list = [p for sub in dataset_path_list_list for p in sub] #same as flatten_list
    norm_stats, all_episode_len = get_norm_stats(dataset_path_list, print, cache_path="data/split_data/mirror_single.json")
    rank0_print(f"{RED}All images: {sum(all_episode_len)}, Trajectories: {len(all_episode_len)}{RESET}")

    train_episode_len_l = [[all_episode_len[i] for i in ids] for ids in train_episode_ids_l]
    val_episode_len_l   = [[all_episode_len[i] for i in ids] for ids in val_episode_ids_l]

    
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]

        # calculate norm stats across all episodes
        # 计算 norm_stats 前，加一行过滤
        stats_paths = flatten_list([
            find_all_hdf5(stats_dir, skip_mirrored_data, rank0_print=rank0_print)
            for stats_dir in stats_dir_l
        ])

        stats_paths = filter_valid_hdf5(stats_paths, rank0_print)   # ← 新增
        norm_stats, _ = get_norm_stats(stats_paths)

    # norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data, rank0_print=rank0_print) for stats_dir in stats_dir_l]))

    # calculate norm stats corresponding to each kind of task
    rank0_print(f'Norm stats from: {[each.split("/")[-1] for each in stats_dir_l]}')
    # rank0_print(f'train_episode_len_l: {train_episode_len_l}') #wzjprint


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
