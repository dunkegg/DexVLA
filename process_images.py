import h5py
import os
import numpy as np
from label.qwen_label import QwenLabeler
import gc
from typing import List
import logging

def generate_annotation_for_image(image_data):
    """
    为每张图片生成标注信息（可以根据图片内容或其他信息生成）。
    
    Args:
        image_data: 图像数据（可以用来生成标注）。
        
    Returns:
        str: 为图片生成的标注。
    """
    # 这里我们简单示范：根据图片的索引生成标注，你可以自定义规则
    return f"Annotation for image with shape {image_data.shape}"

def qwen_lable(h5_file_path, labeller):
    """
    读取 HDF5 文件，并根据图片生成标注信息，最后将标注信息保存到文件中。
    
    Args:
        h5_file_path (str): 需要编辑的 HDF5 文件路径。
        
    Returns:
        None
    """
    print(f"label {h5_file_path} ")
    with h5py.File(h5_file_path, "a") as f:
        # 新的标注数据集的名称
        annotations_dataset_name = 'annotations_long2'
        if annotations_dataset_name in f:
            annotations = f[annotations_dataset_name][()]
            if annotations[0].decode('utf-8')  != '':
                logging.info("skip")
                return False
        # logging.info(f"label {h5_file_path} ")
        
        # 读取现有的图片数据集
        images = f['obs'][:]

        # if len(images)<=43:
        #     return False
        
        instruction = f['instruction'][()].decode('utf-8') 
        # positions = List(f['abs_pos'][:])
        # 为每张图片生成标注
        new_annotations = [''] *  len(images)

        times = 0
        while times<5:
            try:
                if len(images)>20:
                    type = "long"
                    new_annotations = labeller.label_images_long(images, instruction, type)
                else:
                    type = "short"
                    new_annotations = labeller.label_images_short(images, instruction, type)

                # type = "long"
                # new_annotations = labeller.label_images_long(images, instruction, type)
                break
            except Exception as e:
                times +=1

        
        # 检查是否已经存在标注数据集，如果存在则删除它
        if annotations_dataset_name in f:
            del f[annotations_dataset_name]
        if "type" in f:
            del f["type"]
        
        # 将新的标注信息添加到 HDF5 文件

        cleaned = [s if s is not None else "null" for s in new_annotations]
        f.create_dataset(annotations_dataset_name, data=np.array(cleaned, dtype=h5py.string_dtype(encoding='utf-8')))
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("type", data=type, dtype=dt)
        logging.info(f"New annotations added to {h5_file_path}")


        f.flush()  # 强制将内存中的数据写入磁盘
        del f  # 删除文件对象，释放内存
        gc.collect()  # 强制进行垃圾回收
        return True

def process_directory(directory, labeller):
    
    """
    遍历目录中的所有 HDF5 文件，并为每个文件添加标注信息。
    
    Args:
        directory (str): 需要遍历的目录路径。
    
    Returns:
        None
    """
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):  # 只处理 .h5 文件
            h5_file_path = os.path.join(directory, filename)
            logging.info(f"Processing {h5_file_path}")
            qwen_lable(h5_file_path, labeller)
            # try:
            #     # 尝试处理文件
            #     qwen_lable(h5_file_path, labeller)
            # except Exception as e:
            #     # 如果发生异常，打印错误信息并继续处理下一个文件
            #     logging.info(f"Error processing {h5_file_path}: {e}")
            #     continue  # 跳过当前文件，继续下一个文件

def main():
    # 示例目录路径
    # directory = "/wangzejin/code/DexVLA/data/hd5f"  # 替换为你的文件目录路径
    directory = "/wangzejin/code/DexVLA/data/hd5f"  # 替换为你的文件目录路径
    logging.basicConfig(filename='debug.log', level=logging.INFO, filemode='w')
    logging.info("Process started")
    labeller = QwenLabeler()
    # 遍历目录并处理每个 HDF5 文件
    process_directory(directory, labeller)

if __name__ == "__main__":
    main()
