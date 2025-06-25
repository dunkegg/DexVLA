import h5py
import numpy as np
import os
from pathlib import Path
import shutil
import zarr
def process_file(s, new_h5_path, cam_name, n_frames=10):
    try:
        with h5py.File(new_h5_path, 'w') as fout:
            # 读取数据
            action = s["action"]
            fout.create_dataset("/action", data=action)

            qpos = s["qpos"]
            fout.create_dataset("/observations/qpos", data=qpos)
            # qpos = np.zeros(3,dtype=float)
            # fout.create_dataset("/observations/qpos", data=qpos, compression='gzip')

            language_raw = s["language_raw"]
            fout.create_dataset("language_raw", data=language_raw)

            history_paths = s["observations/history_images"][()]

            # 处理历史图片路径
            if int(s["obs_idx"][()]) <= n_frames:  # 临时条件
                if len(history_paths) > 1:
                    history_paths[0] = history_paths[1]
                else:
                    history_paths[0] = s["observations/images"][()]
                history_paths = history_paths.tolist()

                # 补齐 history_paths 长度
                if n_frames > len(history_paths):
                    history_paths = [history_paths[0]] * (n_frames - len(history_paths)) + history_paths

            fout.create_dataset("/observations/history_images", data=np.array(history_paths), compression='gzip')

            obs = s["observations/images"]
            fout.create_dataset(f"/observations/images/{cam_name}", data=obs)

    except Exception as e:
        print(f"Error while processing {new_h5_path}: {e}")
        # 删除已经创建的部分文件，确保文件系统干净
        if os.path.exists(new_h5_path):
            os.remove(new_h5_path)
        print(f"File {new_h5_path} has been removed due to the error.")
        return False  # 返回 False 表示文件处理失败
    return True  # 返回 True 表示文件处理成功
import numcodecs

def process_file_zarr(s, new_zarr_path, cam_name, n_frames=10):
    try:
        # 如果已经存在旧的 zarr 文件，先清除（确保干净）
        if os.path.exists(new_zarr_path):
            shutil.rmtree(new_zarr_path)

        root = zarr.open(new_zarr_path, mode='w')

        # 写入 action
        action = s["action"]
        root.create_dataset("/action", data=action, compressor=zarr.Blosc(cname='zstd', clevel=3))

        # 写入 qpos
        qpos = s["qpos"]
        root.create_dataset("/observations/qpos", data=qpos, compressor=zarr.Blosc(cname='zstd', clevel=3))

        # 写入 language_raw
        # language_raw = s["language_raw"]
        # root.create_dataset("language_raw", data=language_raw)
        language_raw = s["language_raw"][()].decode("utf-8")

        # language_raw = np.array(language_raw, dtype=object)
        root.create_dataset(
            "language_raw",
            data=language_raw,
            compressor=zarr.Blosc(cname='zstd', clevel=3),
            object_codec=numcodecs.VLenUTF8()
        )
        # 处理 history_paths
        history_paths = s["observations/history_images"][()]
        history_paths = [path.decode("utf-8") for path in history_paths]
        if int(s["obs_idx"][()]) <= n_frames:
            if len(history_paths) > 1:
                history_paths[0] = history_paths[1]
            else:
                history_paths[0] = s["observations/images"][()]

            if n_frames > len(history_paths):
                history_paths = [history_paths[0]] * (n_frames - len(history_paths)) + history_paths

        # root.create_dataset("/observations/history_images", data=np.array(history_paths), compressor=zarr.Blosc(cname='zstd', clevel=3))


        root.create_dataset(
            "/observations/history_images",
            data=np.array(history_paths, dtype=object),
            compressor=zarr.Blosc(cname='zstd', clevel=3),
            object_codec=numcodecs.VLenUTF8()  # 可变长度 UTF8 字符串
            )



        # 写入当前观测图像
        obs = s["observations/images"][()].decode("utf-8")
        # root.create_dataset(f"/observations/images/{cam_name}", data=obs, compressor=zarr.Blosc(cname='zstd', clevel=3))
        root.create_dataset(
            f"/observations/images/{cam_name}",
            data=obs,
            compressor=zarr.Blosc(cname='zstd', clevel=3),
            object_codec=numcodecs.VLenUTF8()
        )
        a = 1

    except Exception as e:
        print(f"Error while processing {new_zarr_path}: {e}")
        if os.path.exists(new_zarr_path):
            shutil.rmtree(new_zarr_path)
            print(f"Directory {new_zarr_path} has been removed due to the error.")
        return False
    return True

def save_selected_keys_as_individual_h5(src_h5_path, dst_dir):
    """
    仅保存指定的 key。
    
    Args:
        src_h5_path (str): 源 HDF5 文件路径。
        dst_dir (str): 目标目录，用于保存新的 HDF5 文件。
        selected_keys (list): 要保存的 key 列表。
    """
    # 打开原始 hdf5 文件
    
    with h5py.File(src_h5_path, 'r') as fin:
        sgrp_all = fin.get("follow_paths")
        
        if not sgrp_all:
            print(f"No 'follow_paths' group found in {src_h5_path}")
            return 0
        count= 0
        # 获取文件名用于创建唯一的文件名
        base_filename = os.path.basename(src_h5_path).replace('.hdf5', '')
        cam_name = 'cam_high'
        # 遍历 'follow_paths' 组中的每个子组
        for sub in sgrp_all:
            s = sgrp_all[sub]  # 获取子组内容

            # 使用时间戳或子组名保证文件名唯一
            # timestamp = int(time.time())  # 获取当前时间戳
            new_h5_filename = f"{base_filename}_{sub}.hdf5"
            new_h5_path = os.path.join(dst_dir, new_h5_filename)
            new_zarr_filename = f"{base_filename}_{sub}.zarr"
            new_zarr_path = os.path.join(dst_dir, new_zarr_filename)

            n_frames = 10
            # 调用处理文件的函数
            # success = process_file(s, new_h5_path, cam_name, n_frames)
            success = process_file_zarr(s, new_zarr_path, cam_name, n_frames)

            if success:
                print(f"Successfully processed {new_h5_path}")
                count+=1
            else:
                print(f"Skipping file {new_h5_path} due to error.")

        return count
def process_all_hdf5_in_directory(src_dir, dst_dir):
    # 获取所有的 hdf5 文件
    h5_files = sorted(Path(src_dir).glob("*.hdf5"))
    
    if not h5_files:
        print(f"No HDF5 files found in {src_dir}")
        return
    
    # 确保输出目录存在
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    sum = 0
    # 逐个处理所有 hdf5 文件
    for h5_file in h5_files:
        print(f"Processing {h5_file}")

        count = save_selected_keys_as_individual_h5(h5_file, dst_dir)
        sum+=count


        print(f"Already has {sum} episodes")
        # if sum > 10000:
        #     break

if __name__ == "__main__":
    # 设置源目录和目标目录
    src_dir = "/wangzejin/code/DexVLA/data/follow_data/proc"  # 当前目录
    dst_dir = "/wangzejin/code/DexVLA/data/follow_data/train_zarr"  # 输出目录
    # 设置要保存的key

    process_all_hdf5_in_directory(src_dir, dst_dir)
