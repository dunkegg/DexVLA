import base64
from io import BytesIO
from PIL import Image
import numpy as np
from collections import defaultdict


def encode_image(image):
    """把图片转成 base64 字符串发给模型"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # 将 numpy 数组转换为 PIL 图像对象

    buffered = BytesIO()
    image.save(buffered, format="PNG")  # 根据需要设置图像格式，这里以 PNG 为例
    img_bytes = buffered.getvalue()

    # 对字节流进行 Base64 编码
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")
    return encoded_image


def batch_image_indices(images_length, window=10, pad=5):
    """
    生成带前后 pad 的滑动窗口 batch,并返回 begin/mid/end 类型。
    返回:list of dict:
    {
        "indices": [...],
        "type": "begin" | "mid" | "end",
        "main_range": (main_start, main_end)
    }
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if pad < 0:
        raise ValueError("pad must be >= 0")

    batches = []
    stride = window

    start = 0
    while start < images_length:
        main_start = start
        main_end = min(start + window, images_length)

        # pad 前后
        pad_start = max(0, main_start - pad)
        pad_end = min(main_end + pad, images_length)

        # 判定 batch 类型
        if main_start == 0:
            batch_type = "begin"
        elif main_end == images_length:
            batch_type = "end"
        else:
            batch_type = "mid"

        batch = {
            "indices": list(range(pad_start, pad_end)),
            "type": batch_type,
            "main_range": (main_start, main_end - 1),
        }

        batches.append(batch)
        start += stride

    return batches


def group_indices_by_string(strings):
    """
    Args:
        strings (List[str]): 一组字符串，例如 new_annotations
    Returns:
        Dict[str, List[int]]: 键是字符串内容，值是它出现的所有索引
    用于 label_images_action()：同类任务的帧会一起送给 Qwen 批处理。
    """
    group_map = defaultdict(list)
    for i, s in enumerate(strings):
        group_map[s].append(i)
    return group_map
