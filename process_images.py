import h5py
import os
import numpy as np
from label.qwen_label import QwenLabeler


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
    with h5py.File(h5_file_path, "a") as f:
        # 读取现有的图片数据集
        images = f['obs'][:]
        instruction = f['instruction'][()].decode('utf-8') 
        # 为每张图片生成标注
        if len(images)>50:
            type = "describe"
        else:
            type = "split"
        new_annotations = labeller.label_images(images, instruction, type)
        
        # 新的标注数据集的名称
        annotations_dataset_name = 'annotations'
        
        # 检查是否已经存在标注数据集，如果存在则删除它
        if annotations_dataset_name in f:
            del f[annotations_dataset_name]
        
        # 将新的标注信息添加到 HDF5 文件
        f.create_dataset(annotations_dataset_name, data=np.array(new_annotations, dtype=h5py.string_dtype(encoding='utf-8')))
        print(f"New annotations added to {h5_file_path}")

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
            print(f"Processing {h5_file_path}")
            qwen_lable(h5_file_path, labeller)
            # try:
            #     # 尝试处理文件
            #     qwen_lable(h5_file_path, labeller)
            # except Exception as e:
            #     # 如果发生异常，打印错误信息并继续处理下一个文件
            #     print(f"Error processing {h5_file_path}: {e}")
            #     continue  # 跳过当前文件，继续下一个文件

def main():
    # 示例目录路径
    directory = "/wangzejin/code/DexVLA/data/sample"  # 替换为你的文件目录路径
    
    labeller = QwenLabeler()
    # 遍历目录并处理每个 HDF5 文件
    process_directory(directory, labeller)

if __name__ == "__main__":
    main()
