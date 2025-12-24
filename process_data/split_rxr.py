import h5py
import numpy as np
import os
from pathlib import Path
import shutil
import zarr
def process_file(s, new_h5_path, cam_name, n_frames=10):
    try:
        with h5py.File(new_h5_path, 'w') as fout:
            fout.attrs["tag"] = "rxr"
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
                    history_paths = history_paths.tolist()
                else:
                    history_paths = []
                    history_paths.append(s["observations/images"][()])
                

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
    return True  # 返回 True 表示文件处理成功W
def save_selected_keys_as_individual_h5(src_h5_path, dst_dir):
    try:
        with h5py.File(src_h5_path, 'r') as fin:
            if "follow_paths" not in fin:
                print(f"[SKIP] No 'follow_paths' group in {src_h5_path}")
                return 0

            sgrp_all = fin["follow_paths"]
            count = 0
            base_filename = os.path.basename(src_h5_path).replace('.hdf5', '')
            cam_name = 'cam_high'

            for sub in sgrp_all:
                s = sgrp_all[sub]

                # 取出 language_raw（可能是 list）
                language_raw_list = s["language_raw"][()]

                # 确保是 list
                if not isinstance(language_raw_list, (list, np.ndarray)):
                    language_raw_list = [language_raw_list]

                # 针对 list 中每个元素分别生成一个 hdf5
                for i, lng in enumerate(language_raw_list):
                    new_h5_filename = f"{base_filename}_{sub}_lng{i}.hdf5"
                    new_h5_path = os.path.join(dst_dir, new_h5_filename)

                    # 使用一个“包装的 group-like dict”，覆盖 language_raw 字段
                    s_override = {
                        "action": s["action"][()],
                        "qpos": s["qpos"][()],
                        "language_raw": np.string_(lng),   # 单个字符串
                        "observations/history_images": s["observations/history_images"],
                        "observations/images": s["observations/images"],
                        "obs_idx": s["obs_idx"][()],
                    }

                    try:
                        # 传入包装后的 s
                        success = process_file(s_override, new_h5_path, cam_name, n_frames=10)
                        if success:
                            count += 1
                        else:
                            print(f"[FAIL] Process error in {new_h5_path}, skipping.")
                    except Exception as e:
                        print(f"[ERROR] Exception processing {new_h5_path}: {e}")
                        continue

            return count
    except OSError as e:
        print(f"[ERROR] Failed to open HDF5: {src_h5_path} — {e}")
        return 0
    except Exception as e:
        print(f"[ERROR] Unexpected error: {src_h5_path} — {e}")
        return 0


def process_all_hdf5_in_directory(src_dir, dst_dir):
    h5_files = sorted(Path(src_dir).glob("*.hdf5"))
    
    if not h5_files:
        print(f"[WARNING] No HDF5 files found in {src_dir}")
        return

    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    total = 0
    for h5_file in h5_files:
        print(f"\n>>> Processing {h5_file}")
        try:
            count = save_selected_keys_as_individual_h5(h5_file, dst_dir)
            total += count
            print(f"Current total episodes: {total}")
        except Exception as e:
            print(f"[ERROR] Skipping {h5_file} due to: {e}")
            continue

if __name__ == "__main__":
    # 设置源目录和目标目录
    src_dir = "./data/proc_data/rxr_new"  # 当前目录
    dst_dir = "./data/split_data/rxr"  # 输出目录
    # 设置要保存的key

    process_all_hdf5_in_directory(src_dir, dst_dir)
