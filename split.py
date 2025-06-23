import h5py
import numpy as np
import os
from pathlib import Path
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
            
            with h5py.File(new_h5_path, 'w') as fout:
                action = s["action"]
                fout.create_dataset("/action", data=action)
                qpos = s["qpos"]
                fout.create_dataset("/observations/qpos", data=qpos)
                language_raw = s["language_raw"]
                fout.create_dataset("language_raw", data=language_raw)
                history_paths = s["observations/history_images"][()]

                n_frames = 10
                if int(s["obs_idx"][()])<=n_frames:  # 临时
                    if len(history_paths)>1:
                        history_paths[0] = history_paths[1]
                    else:
                        history_paths[0] = s["observations/images"][()]
                    history_paths = history_paths.tolist()
                    if n_frames > len(history_paths):
                        history_paths = [history_paths[0]] * (n_frames - len(history_paths)) + history_paths

                fout.create_dataset("/observations/history_images", data=np.array(history_paths), compression='gzip')

                obs = s["observations/images"]
                fout.create_dataset(f"/observations/images/{cam_name}", data=obs)

                # print(f"Saved selected keys from {sub} to {new_h5_path}")
        
        return len(sgrp_all)

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
        times = 3
        while times > 0:
            try:
                count = save_selected_keys_as_individual_h5(h5_file, dst_dir)
                sum+=count
                break
            except KeyError as e:
                print(f"KeyError while processing: {e}")
                times-=1
                continue  # 跳过当前子文件
            except Exception as e:
                print(f"Unexpected error while processing: {e}")
                times-=1
                continue  # 跳过当前子文件
        print(f"Already has {sum} episodes")
        if sum > 50000:
            break

if __name__ == "__main__":
    # 设置源目录和目标目录
    src_dir = "/wangzejin/code/DexVLA/data/follow_data/proc"  # 当前目录
    dst_dir = "/wangzejin/code/DexVLA/data/follow_data/train_hdf5"  # 输出目录
    # 设置要保存的key

    process_all_hdf5_in_directory(src_dir, dst_dir)
