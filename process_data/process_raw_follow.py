#!/usr/bin/env python3
"""process_hdf5.py  ▸  Convert saved episode HDF5  ➜  PNG frames + light‑weight HDF5

兼容 **save_output_to_h5()** 中的保存格式：
    • /obs/color_0_0            or /obs 单 dataset
    • /follow_paths/000000/{
          obs_idx          (int32)
          follow_pos       (3,)
          follow_quat      (4,)
          follow_yaw       (float)
          human_pos        (3,)
          human_quat       (4,)
          human_yaw        (float)
          rel_path         (N,8)
      }

本脚本做：
1. 将 /obs/... dataset 中的每帧图像导出为 PNG (frame_000000.png …)
2. 为每条 follow_path 生成：
      observation          = PNG 路径（当前帧）
      history_observation  = 前 5 帧 PNG 路径 list
      rel_path             = 原 rel_path 但位置已减去 follow_pos.x/y
3. 把以上信息写入新的 HDF5，结构：
      /frame_paths               (N, ) string  —— 所有 PNG 路径
      /follow_paths/000000/{ observation, history_observation, rel_path, follow_* , human_* }

Usage:
    python process_hdf5.py episode.hdf5  frames_dir  episode_processed.hdf5
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import magnum as mn
import math


def wrap_pi(angle):
    """把任意角度包到 (-π, π] 区间"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def world2local(rel_path: np.ndarray,
                human_state:  np.ndarray,
                follow_quat: mn.Quaternion,
                follow_yaw: float,
                type: int) -> np.ndarray:
    """
    rel_path : (N, 8)  已经做过 位置减 follow_pos
               列序 [x y z w x y z yaw_world]
    follow_quat : mn.Quaternion  (w, xyz)  用于旋转向量
    follow_yaw  : float (rad)    已有的朝向标量
    """
    R_inv = follow_quat.inverted()          # q_f⁻¹
    out = rel_path.copy()

    human_local = R_inv.transform_vector(mn.Vector3(*human_state))
    # 1) 位置向量旋转到局部系
    for i, v in enumerate(rel_path[:, :3]):
        v_local = R_inv.transform_vector(mn.Vector3(*v))
        if type ==1:
            # out[i, :3] = [-v_local.x, v_local.y, v_local.z]
            out[i, :3] = [v_local.x, v_local.y, -v_local.z]
        else:
            out[i, :3] = [-v_local.x, v_local.y, -v_local.z]

    # 2) 四元数旋转到局部系
    for i, q in enumerate(rel_path[:, 3:7]):
        q_world = mn.Quaternion(mn.Vector3(q[1:]), q[0])
        q_local = R_inv * q_world
        out[i, 3:7] = [q_local.scalar,
                       q_local.vector.x,
                       q_local.vector.y,
                       q_local.vector.z]

    # 3) yaw 差值
    out[:, 7] = wrap_pi(rel_path[:, 7] - follow_yaw)
    if type ==1:
        # human_local =  [-human_local.x, human_local.y, human_local.z]
        human_local =  [human_local.x, human_local.y, -human_local.z]
    else:
        human_local =  [-human_local.x, human_local.y, -human_local.z]
    
    return out.astype(np.float32),human_local

def world2local_target(
                human_state:  np.ndarray,
                follow_quat: mn.Quaternion,
                type: int) -> np.ndarray:
    """
    rel_path : (N, 8)  已经做过 位置减 follow_pos
               列序 [x y z w x y z yaw_world]
    follow_quat : mn.Quaternion  (w, xyz)  用于旋转向量
    follow_yaw  : float (rad)    已有的朝向标量
    """
    R_inv = follow_quat.inverted()          # q_f⁻¹


    human_local = R_inv.transform_vector(mn.Vector3(*human_state))

    if type ==1:
        # human_local =  [-human_local.x, human_local.y, human_local.z]
        human_local =  [human_local.x, human_local.y, -human_local.z]
    else:
        human_local =  [-human_local.x, human_local.y, -human_local.z]
    
    return human_local
def save_depth_png(depth_m, path):
    """
    depth_m: float32, 单位 米
    保存为 uint16 PNG，单位 毫米
    """
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)
def save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2 or arr.shape[-1] == 1:  # 灰度
        Image.fromarray(arr.squeeze()).save(path)
    else:
        Image.fromarray(arr[..., :3]).save(path)

def load_obs_frames(h5: h5py.File, sensor: str):
    """
    sensor: "rgb" | "depth"
    return: List[np.ndarray]  按时间排序
    """
    if "obs" not in h5 or sensor not in h5["obs"]:
        raise KeyError(f"找不到 /obs/{sensor}")

    grp = h5["obs"][sensor]

    # 按 key 排序，确保时间顺序
    keys = sorted(grp.keys(), key=lambda x: int(x))
    frames = [grp[k][()] for k in keys]

    return frames
def locate_obs_dataset(h5: h5py.File):
    """Return (dataset, key). 支持 /obs dataset 或 /obs/color_0_0 等"""
    if "obs" not in h5:
        raise KeyError("文件里找不到 obs 组")
    node = h5["obs"]
    if isinstance(node, h5py.Dataset):
        return node, "obs"
    # group: 找 color_0_0，否则第一个
    if "color_0_0" in node:
        return node["color_0_0"], "color_0_0"
    first = next(iter(node.keys()))
    return node[first], first

def interpolate_rel_path(rel_path: np.ndarray,
                         chunk_size: int,
                         max_dist: float) -> np.ndarray:
    """
    把 (x,z,yaw) 路径插值 / 截断到固定长度 chunk_size.
    rel_path : (...,8) 或 (...,3)
    """
    if rel_path.ndim != 2 or rel_path.shape[1] not in (3, 8):
        raise ValueError("rel_path shape must be (N,3) or (N,8)")

    # 1. 取 x,z,yaw
    data = rel_path[:, [0, 2, 7]] if rel_path.shape[1] == 8 else rel_path.copy()

    # 2. 特例：全 0 或空
    if data.size == 0 or np.allclose(data, 0):
        return np.zeros((chunk_size, 3), np.float32)

    # 3. 计算沿线累积距离
    diffs  = np.diff(data[:, :2], axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    s_full = np.concatenate(([0], np.cumsum(dists)))        # len = N
    total  = s_full[-1]

    # 若超过 max_dist → 找截断点 (总长≥max_dist)
    if total > max_dist:
        idx = np.searchsorted(s_full, max_dist)
        if idx == len(s_full):
            idx -= 1
        # 截断为 idx+1 个点，并在 idx 点上插入精确 max_dist 位置
        excess = s_full[idx] - max_dist
        if excess > 1e-6 and idx > 0:
            ratio = (dists[idx-1] - excess) / dists[idx-1]
            interp_pt = data[idx-1] + ratio * (data[idx] - data[idx-1])
            data = np.vstack([data[:idx], interp_pt])
            s_full = np.concatenate(([0], np.cumsum(np.linalg.norm(
                np.diff(data[:, :2], axis=0), axis=1))))

        total = max_dist

    # 4. 等间距采样到 chunk_size
    if chunk_size == 1:
        samples = np.array([[0, 0, 0]], np.float32)
    else:
        s_samples = np.linspace(0, total, chunk_size)
        samples = np.zeros((chunk_size, 3), np.float32)
        yaw_src = np.unwrap(data[:, 2])    
        for k, s in enumerate(s_samples):
            idx = np.searchsorted(s_full, s) - 1
            idx = np.clip(idx, 0, len(s_full) - 2)
            seg_len = s_full[idx+1] - s_full[idx]
            if seg_len < 1e-8:
                samples[k] = data[idx]
            else:
                t = (s - s_full[idx]) / seg_len
                samples[k, :2] = data[idx, :2] + t * (data[idx+1, :2] - data[idx, :2])
                # yaw 带环绕，简单线插足够（前提 Δyaw 不跨 ±π）
                # samples[k, 2] = data[idx, 2] + t * (data[idx+1, 2] - data[idx, 2])
                yaw_lin = yaw_src[idx] + t * (yaw_src[idx+1] - yaw_src[idx])   # ② 线性插值
                samples[k, 2] = (yaw_lin + np.pi) % (2 * np.pi) - np.pi   

    return samples.astype(np.float32)


# ----------------------------------------------------------- main --------- #
def is_valid_image(rgb: np.ndarray, img_width: int, img_height: int, threshold: float = 0.2):
    """判断一张图是否有效（非纯黑像素占比足够高）"""
    num_black_pixels = np.sum(np.sum(rgb, axis=-1) == 0)
    return num_black_pixels < threshold * img_width * img_height

def check_episode_validity(obs_ds, max_check_frames=3, threshold: float = 0.2):
    """检查前 max_check_frames 帧是否有效（大面积黑图则无效）"""
    for i in range(min(max_check_frames, len(obs_ds))):
        rgb = obs_ds[i]
        height, width = rgb.shape[:2]  # 自动读取图像高宽
        rgb3 = rgb[..., :3]  # 只取前三通道
        num_black_pixels = np.sum(np.all(rgb3 == 0, axis=-1))
        # num_black_pixels = np.sum(np.sum(rgb, axis=-1) == 0)
        if num_black_pixels >= threshold * width * height:
            return False  # 当前帧是大面积黑图
    return True

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

def process_one(src_file: Path, frames_root: Path, dst_root: Path, viz_root: Path|None, hist: int):
    ep_name = src_file.stem              # episode_000 etc.
    frames_dir = frames_root/ep_name
    dst_h5    = dst_root / f"{ep_name}_proc.hdf5"
    # if dst_h5.exists():
    #     print(f"⚠️  Skip {ep_name} — already exists: {dst_h5}")
    #     return
    count = 0
    with h5py.File(src_file,"r") as fin:
        # obs_ds,_ = locate_obs_dataset(fin)
        rgb_frames   = load_obs_frames(fin, "rgb")     # List[(H,W,3)]
        depth_frames = load_obs_frames(fin, "depth")   # List[(H,W)]
        if not check_episode_validity(rgb_frames):
            print(f"❌ {src_file} 前几帧无效图像，跳过")
            return 0         # 跳过该 episode
        rgb_frame_paths=[]
        for i in range(len(rgb_frames)):
            png = frames_dir / f"rgb_{i:06d}.png"
            if not png.exists():
                save_png(rgb_frames[i], png)
            rel_path = png.as_posix()
            rgb_frame_paths.append(rel_path)
        depth_frame_paths=[]
        for i in range(len(depth_frames)):
            png = frames_dir / f"depth_{i:06d}.png"
            if not png.exists():
                save_depth_png(depth_frames[i], png)
            rel_path = png.as_posix()
            depth_frame_paths.append(rel_path)
        with h5py.File(dst_h5,"w") as fout:
            # cam_name = 'cam_high'
            fout.create_dataset("rgb_frame_paths",data=np.array(rgb_frame_paths,dtype=h5py.string_dtype()))
            fout.create_dataset("depth_frame_paths",data=np.array(depth_frame_paths,dtype=h5py.string_dtype()))
            sgrp_all=fin["follow_paths"]; dgrp_all=fout.create_group("follow_paths")
            for sub in sgrp_all:
                s=sgrp_all[sub]; d=dgrp_all.create_group(sub)
                obs_idx=int(s["obs_idx"][()])
                if obs_idx == 0:
                    continue

                type = 0
                # if type == 1:
                #     type = 0
                # else:
                #     print(f"type: {type}")
                human_description = s.get("desc", None)
                if human_description:
                    instruction = "Follow  " + human_description[()].decode("utf-8")
                else:
                    instruction = "Follow the human."
                d.create_dataset("obs_idx",data=obs_idx)
                d.create_dataset(f"observations/images",data=rgb_frame_paths[obs_idx],dtype=h5py.string_dtype())
                start=max(0,obs_idx-hist);hist_paths=rgb_frame_paths[start:obs_idx]
                d.create_dataset("observations/history_images",data=np.array(hist_paths,dtype=h5py.string_dtype()))
                # for key in ("follow_pos","follow_quat","follow_yaw","human_pos","human_quat","human_yaw"):
                #     d.create_dataset(key,data=s[key][()])
                # print(obs_idx)
                # print(s["follow_yaw"][()])
                raw_path=s["rel_path"][()].astype(np.float32)
                fx,fy, fz=s["follow_pos"][()]
                human_pos = s["human_pos"][()]
                human_pos[0] -=fx; human_pos[1]-=fy; human_pos[2]-=fz
                follow_quat =  mn.Quaternion(mn.Vector3(s["follow_quat"][()][1:]), s["follow_quat"][()][0])
                follow_yaw  = s["follow_yaw"][()]
                raw_path[:,0]-=fx; raw_path[:,1]-=fy; raw_path[:,2]-=fz
                reletive_path, huamn_local  = world2local(raw_path,human_pos, follow_quat, follow_yaw,type)
                
                
                # d.create_dataset("rel_path",data=reletive_path)
                actions = interpolate_rel_path(reletive_path, 30, 3.0)
                actions = smooth_yaw(actions,5)
                d.create_dataset('language_raw', data=instruction)
                d.create_dataset('action', data=actions, compression='gzip')
                qposes = np.zeros_like(actions)
                d.create_dataset('qpos', data=qposes, compression='gzip')
                

                llm_action = 'STOP'
                if s.attrs.get("pixel_coords_is_none", False):
                    pixel_coords = None
                else:
                    pixel_coords = s["pixel_coords"][()]
                    u, v = int(pixel_coords[0]), int(pixel_coords[1])
                    pixel_str = f"({u},{v})"
                    llm_action= pixel_str
                substep_reasonings = np.array([llm_action] * len(actions), dtype=object)
                d.create_dataset("substep_reasonings",
                                      data=np.array(substep_reasonings, dtype=h5py.string_dtype()))
                
                count += 1
                if viz_root and False:
                    visualize_follow_path(d, actions,huamn_local, viz_root/ep_name/f"action_{obs_idx}_{type}.png")
    print(f"✓ {ep_name} -> {dst_h5}")
    return count

# def main(src_dir: Path, frames_dir: Path, dst_dir: Path, viz_dir: Path|None, history:int):
#     src_dir, frames_dir, dst_dir = map(Path,(src_dir,frames_dir,dst_dir))
#     frames_dir.mkdir(parents=True,exist_ok=True)
#     dst_dir.mkdir(parents=True,exist_ok=True)

#     h5_files=sorted(src_dir.glob("*.hdf5"))
#     if not h5_files:
#         print("‼ 未找到 *.hdf5 文件于",src_dir); return

#     count = 0
#     for f in h5_files:
#         count += process_one(f, frames_dir, dst_dir, viz_dir, history)
#         print(f"Processed {count} cases in total.")
import os
import multiprocessing as mp
from pathlib import Path
from typing import Optional

def get_io_optimal_workers(num_tasks: int) -> int:
    """
    专门为I/O密集型任务优化的进程数计算
    保存图片主要是磁盘I/O，进程在等待磁盘时CPU是空闲的
    """
    cpu_cores = mp.cpu_count()
    
    # I/O密集型任务：可以设置更多进程
    # 因为大部分时间在等待I/O，而不是使用CPU
    if num_tasks <= cpu_cores * 2:
        # 任务不多时，用CPU核心数的2-4倍
        return min(num_tasks, cpu_cores * 4)
    else:
        # 任务很多时，但不要无限制增加
        # 太多进程会导致磁盘I/O竞争，反而变慢
        return min(
            cpu_cores * 4,  # 通常2-4倍CPU核心数
            1,  # 最大限制，避免过多进程竞争磁盘
            num_tasks
        )

def main(src_dir: Path, frames_dir: Path, dst_dir: Path, 
         viz_dir: Optional[Path], history: int):
    
    h5_files = sorted(src_dir.glob("*.hdf5"))
    if not h5_files:
        print(f"‼ 未找到 *.hdf5 文件于 {src_dir}")
        return
    
    # 针对I/O密集型任务优化
    num_workers = get_io_optimal_workers(len(h5_files))
    
    print(f"CPU核心数: {mp.cpu_count()}")
    print(f"h5文件数: {len(h5_files)}")
    print(f"使用进程数: {num_workers} (I/O密集型任务)")
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        # 使用异步提交，提高并发度
        results = []
        tasks = [(f, frames_dir, dst_dir, viz_dir, history) for f in h5_files]
        
        # 异步提交所有任务
        async_results = [
            pool.apply_async(process_one, task)
            for task in tasks
        ]
        
        # 收集结果
        for i, async_result in enumerate(async_results, 1):
            result = async_result.get()  # 会阻塞直到任务完成
            results.append(result)
            print(f"已完成 {i}/{len(h5_files)} 个文件，累计处理 {sum(results)} 个案例")
    
    total_cases = sum(results)
    print(f"处理完成！总共处理了 {total_cases} 个案例")
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="batch convert episodes")
    
    args=ap.parse_args()
    args.src_dir = "data/raw_data/raw_single_follow_data"
    args.frames_dir ="data/frames/single_follow"
    args.dst_dir = "data/proc_data/single_follow"

    args.src_dir = "data/raw_data/single_follow_pixel"
    args.frames_dir ="data/frames/single_follow_pixel"
    args.dst_dir = "data/proc_data/single_follow_pixel"

    args.viz = "results/multi_follow"
    args.history = 10
    viz=Path(args.viz) if args.viz else None
    main(Path(args.src_dir), Path(args.frames_dir), Path(args.dst_dir), viz, args.history)

