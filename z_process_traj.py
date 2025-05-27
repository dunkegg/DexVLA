import os
import h5py
import numpy as np
import cv2

def process_hdf5_files(hdf5_dir):
    # 遍历文件夹中的所有HDF5文件
    for filename in os.listdir(hdf5_dir):
        if filename.endswith(".h5"):
            file_path = os.path.join(hdf5_dir, filename)

            # 打开HDF5文件
            try:
                with h5py.File(file_path, 'r+') as f:
                    rel_pos = f['relative_pos'][:]  # 位置数据，形状 (T, 3)
                    rel_yaw = f['relative_yaw'][:]  # 旋转数据，形状 (T, 3)




                    # 仅保留第 1 列和第 3 列，并保留两位小数
                    # rel_pos = rel_pos[:, [0, 2]]
                    rel_pos = -np.round(rel_pos, 2)

                    if 'relative_pos_new' in f:
                        del f['relative_pos_new']
                    if 'relative_pos_new' in f:
                        del f['relative_yaw_new'] 

                    f.create_dataset("relative_pos_new", data=rel_pos, compression="gzip")
                    f.create_dataset("relative_yaw_new", data=rel_yaw, compression="gzip")
                        


            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


# 设定你的 HDF5 文件夹路径
hdf5_dir = 'data/hd5f'

# 调用函数处理 HDF5 文件
process_hdf5_files(hdf5_dir)
