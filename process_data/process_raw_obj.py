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
def world2local_path(rel_path: np.ndarray,
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

    return out.astype(np.float32)
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

# ----------------------------------------------------------- utilities ---- #

def save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2 or arr.shape[-1] == 1:  # 灰度
        Image.fromarray(arr.squeeze()).save(path)
    else:
        Image.fromarray(arr[..., :3]).save(path)


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


# def visualize_follow_path(group: h5py.Group, actions,human_local, out_png: Path):
#     """
#     两张图：
#       • 上：rel_path 在 X-Z 平面（绿色折线）+ follow/human 星标
#       • 下：Δyaw 随时间步的变化（角度 °）
#     """
#     rel_path = group["rel_path"][()]
#     # yaw_deg  = np.degrees(rel_path[:, 7])          # rad → deg
#     yaw_deg = np.degrees(actions[:2])
#     steps    = np.arange(len(rel_path))

#     fig, (ax_top, ax_bot) = plt.subplots(
#         2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [2, 1]}
#     )

#     # ── 上图：轨迹 ────────────────────────────────────────────────
#     # ax_top.plot(rel_path[:, 0], rel_path[:, 2], "g-", label="rel_path X-Z")
#     ax_top.plot(actions[:, 0], actions[:, 1], "g-", label="rel_path X-Z")
#     ax_top.scatter(0, 0, c="red", marker="*", s=100, label="follow (0,0)")
#     ax_top.scatter(human_local[0], human_local[2],
#                    c="blue", marker="*", s=100, label="human_local")
#     ax_top.set_aspect("equal"); ax_top.set_xlim(-5, 5); ax_top.set_ylim(-5, 5)
#     ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("z (m)")
#     ax_top.legend(fontsize="small")
#     ax_top.set_title(f"obs_idx = {int(group['obs_idx'][()])}")

#     # ── 下图：Δyaw-timestep ─────────────────────────────────────
#     ax_bot.plot(steps, yaw_deg, "m--", lw=1.2)
#     ax_bot.set_xlabel("timestep")
#     ax_bot.set_ylabel("Δyaw (deg)")
#     ax_bot.grid(True, alpha=0.3)

#     fig.tight_layout()
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png, dpi=140)
#     plt.close(fig)

from matplotlib.collections import LineCollection
from matplotlib import cm, colors

def visualize_follow_path(group: h5py.Group,
                          actions: np.ndarray,
                          out_png: Path,
                          cmap_name: str = "viridis"):

    traj_xz = actions[:, :2]
    yaw_deg = np.degrees(actions[:, 2])
    steps   = np.arange(len(actions))

    cmap   = cm.get_cmap(cmap_name)
    norm   = colors.Normalize(vmin=0, vmax=len(actions)-1)
    colors_arr = cmap(norm(steps))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=False
    )

    # ── 读取 language_raw（数组）────────────────────────────
    if "language_raw" in group:
        # group["language_raw"][()] 返回 ndarray(shape=(N,))
        language_str = group["language_raw"][()].decode("utf-8")
        # bytes → str
    else:
        language_str = "(no language_raw)"

    # ── 上图：轨迹 ───────────────────────────────────────
    segs = np.concatenate([traj_xz[:-1, None, :], traj_xz[1:, None, :]], axis=1)
    lc   = LineCollection(segs, colors=colors_arr[:-1], linewidths=2)
    ax_top.add_collection(lc)
    ax_top.scatter(0, 0, c="red", marker="*", s=100, label="follow (0,0)")

    ax_top.set_aspect("equal")
    ax_top.set_xlim(-5, 5); ax_top.set_ylim(-5, 5)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("z (m)")
    ax_top.legend(fontsize="small")

    # 以前 title 是 obs_idx，现在加上 language_raw
    obs_idx_val = int(group["obs_idx"][()])
    ax_top.set_title(f"obs_idx = {obs_idx_val}")

    # ── 下图：yaw 曲线 ───────────────────────────────────
    for i in range(len(yaw_deg)-1):
        ax_bot.plot(steps[i:i+2], yaw_deg[i:i+2],
                    color=colors_arr[i], linewidth=2)
    ax_bot.set_xlabel("timestep")
    ax_bot.set_ylabel("Δyaw (deg)")
    ax_bot.grid(True, alpha=0.3)

    # ── 在图像顶部添加文本（全宽，可换行）────────────────────
    fig.text(
        0.5, 0.97,
        language_str,
        ha="center", va="top",
        fontsize=10,
        wrap=True  # 自动换行
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_top, ax_bot], orientation="vertical",
                 fraction=0.03, pad=0.02)

    fig.tight_layout(rect=[0, 0, 1, 0.93])  # 给顶部文字留空间
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

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


def extract_future_segment(rel_path, obs_idx, future_dist):
    """
    从 rel_path 中 obs_idx 开始往后走，直到累计距离达到 future_dist 米
    返回短轨迹 short_path
    """
    pts = rel_path[:, :3]  # 取 xyz
    T = len(pts)

    short = [rel_path[obs_idx]]

    dist_sum = 0.0
    for i in range(obs_idx, T-1):
        p = pts[i]
        q = pts[i+1]
        dist_sum += np.linalg.norm(q - p)
        short.append(rel_path[i+1])
        if dist_sum >= future_dist:
            break

    return np.array(short, dtype=np.float32)

def process_one(src_file: Path, frames_root: Path, dst_root: Path, viz_root: Path|None, hist: int,
                sample_stride=5, future_dist=2.0):
    """
    sample_stride:  每隔多少帧采一个训练样本（例如 5）
    future_dist:    短轨迹的未来距离（单位：米）
    """
    ep_name = src_file.stem
    frames_dir = frames_root/ep_name
    dst_h5    = dst_root / f"{ep_name}_proc.hdf5"
    count = 0

    with h5py.File(src_file, "r") as fin:
        object_category = fin.get("object_category", None)
        object_environment = fin.get("object_environment", None)
        has_obj = object_category is not None and object_environment is not None
        if not has_obj:
            return 0
        # 读取连续轨迹 rel_path_org = [T, 8]
        rel_path_org = fin["rel_path"][()].astype(np.float32)
        T = len(rel_path_org)

        # 检查图像数据
        obs_ds, _ = locate_obs_dataset(fin)
        if not check_episode_validity(obs_ds):
            print(f"❌ {src_file} 前几帧无效图像，跳过")
            return 0

        # ① 保存 PNG 图像
        frame_paths=[]
        for i in range(T):
            png = frames_dir / f"frame_{i:06d}.png"
            if not png.exists():
                save_png(obs_ds[i], png)
            frame_paths.append(png.as_posix())

        # ② 写入 HDF5 输出
        with h5py.File(dst_h5, "w") as fout:
            fout.create_dataset("frame_paths", data=np.array(frame_paths, dtype=h5py.string_dtype()))

            dgrp_all = fout.create_group("follow_paths")

            # 遍历所有采样点
            for obs_idx in range(0, T, sample_stride):

                # ---------- 取历史图像 ----------
                start = max(0, obs_idx - hist)
                hist_paths = frame_paths[start:obs_idx]

                # ---------- 生成短轨迹 ----------
                short_path = extract_future_segment(rel_path_org, obs_idx, future_dist)

                if len(short_path) < 2:
                    continue  # 不够未来数据就跳过

                # ---------- 生成动作 ----------
                fx,fy, fz = short_path[0][:3]
                short_path[:,0]-=fx; short_path[:,1]-=fy; short_path[:,2]-=fz 
                start_yaw = short_path[0][7]
                start_quat = short_path[0][3:7]
                start_quat =  mn.Quaternion(mn.Vector3(start_quat[1:]), start_quat[0])
                reletive_path  = world2local_path(short_path, start_quat, start_yaw,type=0)
                actions = interpolate_rel_path(reletive_path, 30, 3.0)
                qposes = np.zeros_like(actions)

                # 写入子 group
                subgrp = dgrp_all.create_group(str(obs_idx))

                subgrp.create_dataset("obs_idx", data=obs_idx)
                subgrp.create_dataset("observations/images", 
                                      data=frame_paths[obs_idx], 
                                      dtype=h5py.string_dtype())
                subgrp.create_dataset("observations/history_images",
                                      data=np.array(hist_paths, dtype=h5py.string_dtype()))

                # subgrp.create_dataset("rel_path", data=short_path)
                subgrp.create_dataset("action", data=actions, compression='gzip')
                subgrp.create_dataset("qpos", data=qposes, compression='gzip')

                # subgrp.create_dataset("language_raw",
                #                       data="Follow the trajectory.",
                #                       dtype=h5py.string_dtype())

                object_category = fin["object_category"][()]     # bytes
                object_category = object_category.decode("utf-8").rstrip("\x00")
                object_environment = fin["object_environment"][()]     # bytes
                object_environment = object_environment.decode("utf-8").rstrip("\x00")

                instruction = f'''Find the {object_environment}. Your action is 'Move' if you already see it. Otherwise 'Rotate'.
                '''
                dt = h5py.string_dtype(encoding='utf-8')
                subgrp.create_dataset("language_raw", data=instruction, dtype=dt)

                substep_reasonings = np.array(["Move"] * len(actions), dtype=object)

                subgrp.create_dataset("substep_reasonings",
                                      data=np.array(substep_reasonings, dtype=h5py.string_dtype()))

                count += 1
                if viz_root and False:
                    visualize_follow_path(subgrp, actions, viz_root/ep_name/f"action_{obs_idx}_{type}.png")

    print(f"✓ {ep_name} -> {dst_h5} ({count} samples)")
    return count


def main(src_dir: Path, frames_dir: Path, dst_dir: Path, viz_dir: Path|None, history:int):
    src_dir, frames_dir, dst_dir = map(Path,(src_dir,frames_dir,dst_dir))
    frames_dir.mkdir(parents=True,exist_ok=True)
    dst_dir.mkdir(parents=True,exist_ok=True)

    h5_files=sorted(src_dir.glob("*.hdf5"))
    if not h5_files:
        print("‼ 未找到 *.hdf5 文件于",src_dir); return

    count = 0
    for f in h5_files:
        count += process_one(f, frames_dir, dst_dir, viz_dir, history,sample_stride=2, future_dist=2)
        print(f"Processed {count} cases in total.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="batch convert episodes")
    
    args=ap.parse_args()


    args.src_dir = "data/raw_data/obj/move_new"
    args.frames_dir ="data/frames/obj_move"
    args.dst_dir = "data/proc_data/obj_move"

    args.viz = "results/obj_move"
    args.history = 10
    viz=Path(args.viz) if args.viz else None
    main(Path(args.src_dir), Path(args.frames_dir), Path(args.dst_dir), viz, args.history)

