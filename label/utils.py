import base64
import os
import glob
from io import BytesIO
from PIL import Image
import numpy as np
# 定义滑动窗口函数，获取连续图像路径
def get_images_from_path(image_dir, window_size=5):
    """
    滑动窗口获取连续的图像路径。
    image_dir: 图像文件所在目录路径
    window_size: 窗口大小
    """
    # 获取指定路径下的所有图像文件
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))  # 这里假设图像是 .png 格式
    images = []
    
    # 获取窗口内连续的图像
    for i in range(len(image_paths) - window_size + 1):
        window_images = image_paths[i:i + window_size]  # 获取窗口内的连续图像
        images.append(window_images)
    
    return images, len(image_paths), image_paths

# 编码图像为 base64
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # 将 numpy 数组转换为 PIL 图像对象
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # 根据需要设置图像格式，这里以 PNG 为例
    img_bytes = buffered.getvalue()
    
    # 对字节流进行 Base64 编码
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")
    return encoded_image

def batch_image_indices(images_length, batch_size=10):
    """
    从 encoded_images 中按批次获取图片的索引，每批次最多 batch_size 个图片。

    Args:
        encoded_images (list): 编码后的图片列表。
        batch_size (int): 每批次获取的图片数量，默认是 10。

    Returns:
        list of list: 每个子列表包含图片的索引。
    """
    indices_batches = []
    indices = list(range(images_length))  # 生成索引列表

    for i in range(0, len(indices), batch_size):
        # 生成每个批次的索引
        indices_batches.append(indices[i:i + batch_size])

    return indices_batches

def create_short_list(long_list, x):
    """
    从长列表中隔 x 帧取一帧，生成短列表。

    Args:
        long_list (list): 长列表，包含从 1 到 40 的元素。
        x (int): 隔多少帧取一帧。

    Returns:
        short_list (list): 生成的短列表，每隔 x 帧取一个元素。
    """
    short_list = long_list[::x]  # 使用切片获取每隔 x 帧的元素
    return short_list

def map_annotations_to_long_list(short_list, long_list, annotations):
    """
    将短列表中的标注映射回长列表，跳过的帧继承前一个标注。

    Args:
        short_list (list): 短列表，包含选中的帧的索引。
        long_list (list): 长列表，包含从 1 到 40 的元素。
        annotations (list): 短列表的标注信息。

    Returns:
        long_list_with_annotations (list): 更新后的长列表，包含标注。
    """
    long_list_with_annotations = [None] * len(long_list)  # 初始化长列表的标注为 None

    short_index = 0
    for i in range(len(long_list)):
        if short_index < len(short_list) and i == short_list[short_index]:
            # 将标注填入当前帧
            long_list_with_annotations[i] = annotations[short_index]
            short_index += 1
        elif i > 0 and long_list_with_annotations[i-1] is not None:
            # 如果当前帧没有标注，则继承前一帧的标注
            long_list_with_annotations[i] = long_list_with_annotations[i-1]

    return long_list_with_annotations