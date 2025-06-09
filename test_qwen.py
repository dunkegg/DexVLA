import h5py
import numpy as np
import cv2
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# # 加载模型
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "checkpoints/qwen2_vl",
#     torch_dtype="auto",
#     device_map={"": 0},   # <<<<<<<< 这里！
# )
# processor = AutoProcessor.from_pretrained("checkpoints/qwen2_vl")

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "checkpoints/Qwen2.5-VL-72B-Instruct",
    torch_dtype="auto",
    device_map={"": 0},   # <<<<<<<< 这里！
)
processor = AutoProcessor.from_pretrained("checkpoints/Qwen2.5-VL-72B-Instruct")
