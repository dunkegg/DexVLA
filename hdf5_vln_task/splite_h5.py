# import os
# import h5py
# import numpy as np
# from PIL import Image

# input_h5 = "/mnt/pfs/3zpd5q/code/zf/data/raw_data/door.h5"
# output_dir = "/mnt/pfs/3zpd5q/code/zf/data/splite_data"
# images_dir = os.path.join(output_dir, "images")
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(images_dir, exist_ok=True)

# def save_image(img_array, save_path):
#     img = Image.fromarray(img_array)
#     img.save(save_path)

# with h5py.File(input_h5, 'r') as f:
#     episodes = list(f.keys())
#     episode_count = 0

#     for ep in episodes:
#         instruction = f[ep]['instruction'][()]
#         if b"behind" in instruction:  # 跳过包含 'behind' 的指令
#             print(f"Skipping {ep} due to 'behind' in instruction.")
#             continue

#         image_data = f[ep]['image_data'][()]            # (T,H,W,3)
#         real_traj = f[ep]['real_trajectory_data'][()]   # (T,4)
#         ref_traj = f[ep]['reference_trajectory_data'][()]  # (T_ref,2)
#         obj_name = f[ep]['obj_name'][()]

#         episode_count += 1
#         new_h5_name = os.path.join(output_dir, f"episode_{episode_count:02d}.hdf5")

#         with h5py.File(new_h5_name, 'w') as hf:
#             # === 1) 保存 action ===
#             hf.create_dataset("action", data=real_traj[:, :3], dtype='float32')

#             # === 2) 保存语言指令 ===
#             hf.create_dataset("language_raw", data=instruction, dtype=h5py.string_dtype())

#             # === 3) 创建 observations group ===
#             obs_grp = hf.create_group("observations")

#             # ✅ 将 qpos 放入 observations 下
#             T = real_traj.shape[0]
#             obs_grp.create_dataset("qpos", data=np.zeros((T, 3), dtype='float32'))

#             # === 4) 保存图像 ===
#             cam_grp = obs_grp.create_group("images")

#             # cam_high 只需占位
#             cam_grp.create_dataset("cam_high", data=b"", dtype=h5py.string_dtype())

#             # 保存历史图像路径
#             history_imgs = []
#             for t in range(T):
#                 img_name = f"episode_{episode_count:02d}_frame_{t:03d}.png"
#                 save_image(image_data[t], os.path.join(images_dir, img_name))
#                 history_imgs.append(os.path.join(images_dir, img_name).encode())
#             obs_grp.create_dataset("history_images", data=np.array(history_imgs), dtype=h5py.string_dtype())

#             # === 5) 保存 reference_trajs ===
#             hf.create_dataset("reference_trajs", data=ref_traj, dtype='float32')

#         print(f"✅ Saved {new_h5_name} with {T} steps.")
######################################

import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# 参数配置
# -----------------------------
input_h5_list = [
    "/mnt/pfs/3zpd5q/code/zf/raw_data/matterport_data/objnav_rotate1.h5",
    "/mnt/pfs/3zpd5q/code/zf/raw_data/matterport_data/objnav_rotate2.h5"
]

output_dir = "/mnt/pfs/3zpd5q/code/zf/train_data/objnav_data/objnav_rotate_5w"
images_dir = os.path.join(output_dir, "images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# -----------------------------
# 工具函数
# -----------------------------
def save_images_parallel(image_data, episode_idx, images_dir, max_workers=8):
    """多线程保存图片并返回路径数组"""
    history_imgs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for t in range(len(image_data)):
            img_name = f"episode_{episode_idx:04d}_frame_{t:03d}.jpg"
            img_path = os.path.join(images_dir, img_name)
            # OpenCV 写图：RGB -> BGR
            futures.append(executor.submit(cv2.imwrite, img_path, cv2.cvtColor(image_data[t], cv2.COLOR_RGB2BGR)))
            history_imgs.append(img_path.encode())
        # 等待所有任务完成
        for _ in as_completed(futures):
            pass
    return np.array(history_imgs, dtype=h5py.string_dtype())

# -----------------------------
# 主处理逻辑
# -----------------------------
episode_count = 0

for input_h5 in tqdm(input_h5_list, desc="Processing input HDF5 files"):
    with h5py.File(input_h5, 'r', libver='latest') as f:
        episodes = list(f.keys())
        for ep in tqdm(episodes, desc=f"Processing {os.path.basename(input_h5)}", leave=False):
            grp = f[ep]

            # === new 读取数据 ===
            image_data = grp["rgb"][()]                      # (T, H, W, 3)
            obj_name = grp["obj_name"][()]
            action = grp["local_traj"][()]                   # (T, 30, 3)
            language_raw = grp["instruction"][()]
            substep_reasonings = grp["substep_reasonings"][()]

            # === old ===
            # image_data = grp["image_data"][()]                      # (T, H, W, 3)
            # obj_name = grp["obj_name"][()]
            # action = grp["action"][()]                   # (T, 30, 3)
            # language_raw = grp["language_raw"][()]
            # substep_reasonings = grp["substep_reasonings"][()]
            # === 保存新的 episode ===
            episode_count += 1
            new_h5_name = os.path.join(output_dir, f"episode_{episode_count:04d}.hdf5")

            with h5py.File(new_h5_name, 'w', libver='latest') as hf:
                hf.create_dataset("action", data=action, dtype='float32')
                hf.create_dataset("language_raw", data=language_raw, dtype=h5py.string_dtype())
                hf.create_dataset("substep_reasonings", data=substep_reasonings, dtype=h5py.string_dtype())
                hf.create_dataset("obj_name", data=obj_name, dtype=h5py.string_dtype())

                obs_grp = hf.create_group("observations")
                T = len(image_data)
                obs_grp.create_dataset("qpos", data=np.zeros((T, 3), dtype='float32'))
                cam_grp = obs_grp.create_group("images")
                cam_grp.create_dataset("cam_high", data=b"", dtype=h5py.string_dtype())

                # 多线程保存图片
                history_imgs = save_images_parallel(image_data, episode_count, images_dir, max_workers=8)
                obs_grp.create_dataset("history_images", data=history_imgs, dtype=h5py.string_dtype())

print(f"\n🎉 All done! Total {episode_count} episodes saved to {output_dir}")

