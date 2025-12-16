import h5py
import numpy as np
import cv2
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import re

def extract_pixel_coords(output_text):
    """
    从大模型输出中提取 (x, y) 像素坐标
    自动处理 list / dict / str
    """

    # ---------- 1️⃣ 统一成字符串 ----------
    if output_text is None:
        return []

    if isinstance(output_text, list):
        # 常见情况：["text ..."]
        if len(output_text) == 0:
            return []
        output_text = output_text[0]

    if isinstance(output_text, dict):
        # 常见字段兜底
        for k in ["generated_text", "text", "content"]:
            if k in output_text:
                output_text = output_text[k]
                break

    if not isinstance(output_text, str):
        raise TypeError(f"extract_pixel_coords expects str, got {type(output_text)}")

    # ---------- 2️⃣ 去掉 <think> ... </think> ----------
    text_wo_think = re.sub(
        r"<think>.*?</think>",
        "",
        output_text,
        flags=re.DOTALL
    )

    # ---------- 3️⃣ 提取 (x, y) ----------
    matches = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text_wo_think)

    coords = [(int(x), int(y)) for x, y in matches]

    return coords

def extract_bboxes(output_text):
    """
    从模型输出中提取 bounding boxes
    返回: List[(x1, y1, x2, y2)]
    """

    # -------- 1. 统一成字符串 --------
    if isinstance(output_text, list):
        output_text = output_text[0]
    if isinstance(output_text, dict):
        output_text = output_text.get("generated_text", "")

    if not isinstance(output_text, str):
        raise TypeError(f"expect str, got {type(output_text)}")

    # -------- 2. 去掉 <think> --------
    text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)

    # -------- 3. 匹配 (x1, y1, x2, y2) --------
    pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
    matches = re.findall(pattern, text)
    bboxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in matches]

    return bboxes

import cv2
import os

def draw_points_on_image(
    img_path: str,
    points,
    save_path: str,
    radius: int = 6,
    color=(0, 0, 255),  # 红色 (BGR)
    thickness: int = -1  # 实心圆
):
    """
    在图片上绘制像素点并保存

    Args:
        img_path: 原始图片路径
        points: [(x, y), ...] 像素坐标
        save_path: 保存的新图片路径
        radius: 点的半径
        color: BGR 颜色
        thickness: -1 表示实心
    """

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    h, w = img.shape[:2]

    for i, (x, y) in enumerate(points):
        # 边界保护
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, thickness)
            cv2.putText(
                img,
                str(i),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        else:
            print(f"⚠️ point {i} out of bounds: {(x, y)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"✅ Saved visualization to {save_path}")


def draw_bboxes_on_image(
    img_path,
    bboxes,
    save_path,
    color=(0, 255, 0),
    thickness=2
):
    """
    img_path: 原图路径
    bboxes: [(x1,y1,x2,y2), ...]
    save_path: 输出路径
    """

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]

    for (x1, y1, x2, y2) in bboxes:
        # 防止越界
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

    return img
# -------------------------------
# 配置区
# -------------------------------

# H5 文件路径
img_path = 'docs/2.jpg'

# 加载模型
model_path = "checkpoints/MiMo-Embodied-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map={"": 0},   # <<<<<<<< 这里！
)
processor = AutoProcessor.from_pretrained(model_path)



device = model.device  # 获取模型所在设备
print(f"✅ 模型加载在设备：{device}")


instruction = "What do you see in these images?"
instruction = "你看到了什么？ 描述你看到的物品并输出图片里对应物品中心点的像素坐标。"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": instruction},
        ],
    }
]

# -------------------------------
# 准备推理 inputs
# -------------------------------

# 文本模板处理
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 图片、视频预处理
image_inputs, video_inputs = process_vision_info(messages)


# 整合inputs
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# -------------------------------

for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(device)

print("🚀 开始推理...")
generated_ids = model.generate(**inputs, max_new_tokens=4096)

# 只取新生成部分
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 解码输出
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
coords = extract_pixel_coords(output_text)
draw_points_on_image(
    img_path=img_path,
    points=coords,
    save_path="docs/pointed2-1.jpg"
)

bboxes = extract_bboxes(output_text)
draw_bboxes_on_image(
    img_path=img_path,
    bboxes=bboxes,
    save_path="docs/pointed2-2.jpg"
)


print("📝 推理输出：")
# print(output_text)

for text in output_text:
    print(text)
