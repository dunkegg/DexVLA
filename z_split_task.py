import os
import h5py
import numpy as np
import cv2
from collections import defaultdict

def group_indices_by_string(strings):
    """
    Args:
        strings (List[str]): 一组字符串，例如 new_annotations
    Returns:
        Dict[str, List[int]]: 键是字符串内容，值是它出现的所有索引
    """
    group_map = defaultdict(list)
    for i, s in enumerate(strings):
        group_map[s].append(i)
    return group_map

def split_h5_file_with_image_paths(hdf5_dir, output_dir, images_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(hdf5_dir):
        if filename.endswith(".h5"):
            file_path = os.path.join(hdf5_dir, filename)

            # 为每个HDF5文件创建一个新的文件夹
            folder_name = filename.split('.')[0]  # 使用文件名（去掉扩展名）作为文件夹名
            folder_path = os.path.join(images_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            with h5py.File(file_path, 'r') as f:
                cam_name = 'cam_high'
                instruction = f['instruction']
                subtasks = f['annotations_long2'][()]   # 假设 'annotations' 存在
                actions = f['/action'][()]# 假设 'type' 存在
                qposes = f['/observations/qpos'][()]
                
                num_samples = []
                groups = group_indices_by_string(subtasks)
                for subtask, indices in groups.items():
                    if len(indices)<3:
                        num_samples.append(indices[0])
                    elif len(indices)<5:
                        num_samples.append(indices[0])
                        num_samples.append(indices[2])
                    elif len(indices)<7:
                        num_samples.append(indices[0])
                        num_samples.append(indices[2])
                        num_samples.append(indices[-3])
                    else:
                        num_samples+=indices[::-2]  # 倒
       
                # 提取图像帧
                images = f['obs'][:]  

                images_path = []

                # 保存每个图像帧到对应的文件夹
                for i, img in enumerate(images):
                    img_filename = f"frame_{i:04d}.png"
                    img_path = os.path.join(folder_path, img_filename)
                    cv2.imwrite(img_path, img)
                    images_path.append(img_path)

                # 创建 txt 文件，记录 instruction 和 new_annotations
                txt_file_path = os.path.join(folder_path, "annotations.txt")
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write(f"Instruction: {instruction}\n")
                    txt_file.write("Subtasks:\n")
                    for subtask in subtasks:
                        txt_file.write(f"{subtask.decode('utf-8')}\n")  # 每个注释单独一行
                


                dt = h5py.string_dtype(encoding='utf-8')
                for i in num_samples:
                    history_indices = [max(0, i - j) for j in reversed(range(10))]  # i-9 到 i，如果不足重复第0帧
                    history_paths = [images_path[idx] for idx in history_indices]
                    # 2. 写入 HDF5，仅保存图像路径
                    h5_path = os.path.join(output_dir, f"{folder_name}_{i:04d}.hdf5")
                    with h5py.File(h5_path, 'w') as fout:
                        fout.create_dataset(f'/observations/images/{cam_name}', data=images_path[i].encode('utf-8'))  # 路径以 bytes 存入
                        history_bytes = [p.encode('utf-8') for p in history_paths]
                        fout.create_dataset('/observations/history_images', data=np.array(history_bytes, dtype=dt), compression='gzip')
                        fout.create_dataset('instruction',data=instruction)
                        fout.create_dataset('language_raw', data=np.array(subtasks[i]))
                        fout.create_dataset('/action', data=actions[i*10], compression='gzip')
                        fout.create_dataset('/observations/qpos', data=qposes[i*10], compression='gzip')
                    
                    print(f"Saved: {h5_path}")


split_h5_file_with_image_paths('data/hdf5', 'data/vln', 'data/images')