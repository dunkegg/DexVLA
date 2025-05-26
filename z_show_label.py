import os
import h5py
import numpy as np
import cv2

def process_hdf5_files(hdf5_dir):
    # 遍历文件夹中的所有HDF5文件
    for filename in os.listdir(hdf5_dir):
        if filename.endswith(".h5"):
            file_path = os.path.join(hdf5_dir, filename)

            # 为每个HDF5文件创建一个新的文件夹
            folder_name = filename.split('.')[0]  # 使用文件名（去掉扩展名）作为文件夹名
            folder_path = os.path.join(hdf5_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # 打开HDF5文件
            try:
                with h5py.File(file_path, 'r+') as f:
                    # 提取 'instruction' 和 'new_annotations' 信息
                    instruction = f['instruction'][()].decode('utf-8') 
                    new_annotations = f['annotations'][()]  # 假设 'annotations' 存在
                    type_data = f['type'][()].decode('utf-8')   # 假设 'type' 存在

                    # 提取图像帧
                    images = f['obs'][:]  # 假设图像数据存储在 'images' 中

                    # 保存每个图像帧到对应的文件夹
                    for i, img in enumerate(images):
                        img_filename = f"frame_{i:04d}.png"
                        img_path = os.path.join(folder_path, img_filename)
                        cv2.imwrite(img_path, img)
                        print(f"保存图像: {img_path}")

                    # 创建 txt 文件，记录 instruction 和 new_annotations
                    txt_file_path = os.path.join(folder_path, "annotations.txt")
                    with open(txt_file_path, 'w') as txt_file:
                        txt_file.write(f"Instruction: {instruction}\n")
                        txt_file.write(f"Type: {type_data}\n")
                        txt_file.write("Annotations:\n")
                        for annotation in new_annotations:
                            txt_file.write(f"{annotation.decode('utf-8')}\n")  # 每个注释单独一行
                        


            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


# 设定你的 HDF5 文件夹路径
hdf5_dir = 'sampled_data'

# 调用函数处理 HDF5 文件
process_hdf5_files(hdf5_dir)
