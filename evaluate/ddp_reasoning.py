import torch
import deepspeed
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import h5py
import numpy as np
import cv2

# -------------------------------
# 配置区
# -------------------------------

# H5 文件路径
h5_path = 'data_nav/longhu_interaction__img_CAM_A_compressed.h5'

# 解压到哪个目录
output_dir = 'unpacked_images'
os.makedirs(output_dir, exist_ok=True)

# 提取的起止帧
start_idx = 378
num_images = 5
end_idx = start_idx + num_images

# -------------------------------
# DeepSpeed 配置（推理）
# -------------------------------
deepspeed_config = {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": True,  # 使用 ZeRO 优化，适用于大型模型
    "fp16": {"enabled": True},  # 开启 fp16 加速
    "inference_batch_size": 1,
    "dtype": "float16",  # 如果你使用的是 FP16 模型
    "steps_per_print": 10,
}

# -------------------------------
# 加载 DeepSpeed 推理模型
# -------------------------------
# 加载 Qwen2 模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "checkpoints/qwen-72",
    torch_dtype="auto",
)

# 使用 DeepSpeed 对模型进行推理初始化
model = deepspeed.init_inference(model, config_params=deepspeed_config)

# 处理器加载
processor = AutoProcessor.from_pretrained("checkpoints/qwen-72")

# -------------------------------
# 从 h5 提取图像并保存
# -------------------------------
with h5py.File(h5_path, 'r') as h5f:
    images = h5f['images'][start_idx:end_idx]
    filenames = [fn.decode() for fn in h5f['filenames'][start_idx:end_idx]]
    labels = [lb.decode() for lb in h5f['labels'][start_idx:end_idx]]
    command = h5f.attrs['command']
    if isinstance(command, bytes):
        command = command.decode()

    print(f"📋 任务指令：{command}")
    print(f"准备提取 {len(images)} 张图，从第 {start_idx} 到第 {end_idx-1}")

    # 保存图片
    for img, filename, label in zip(images, filenames, labels):
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"✅ 保存 {filepath}，标签：{label}")

# 获取保存后的图片路径
image_paths = [os.path.join(output_dir, filename) for filename in filenames]

# -------------------------------
# 构建推理 messages
# -------------------------------
instruction = "Your task is to go along the road. Please consider if you need to go round obstacles(dynamic or static). Analyse the scenario first then tell what action you should do for the task."

instruction = f"""
You are navigating along a road in a residential area. Your goal is to move forward along the road as smoothly as possible.

You are provided with a sequence of images:
- The first five images represent your recent movement trajectory.
- The last image represents your current view.

Your task:
1. Identify all potential obstacles in the scene.
2. Analyze the spatial relationship between each obstacle and the center of the road.
3. Specifically check whether any object partially or fully overlaps with the centerline of the road.
4. If there is any overlap, treat it as a blocking obstacle and plan an avoidance action.
5. If there is no overlap, it is safe to move forward.

Your output must include two parts:
- Reasoning: Clearly explain which objects exist, their location relative to the centerline, and whether they block forward movement.
- Action: Provide the next action (e.g., 'move forward', 'move slightly left and forward', or 'go around the obstacle to the right and continue').

**Important:**  
- You must not assume the road is clear unless you explicitly confirm that no objects overlap the centerline.
- Visual perception errors must be minimized by strictly basing reasoning only on the visual observations provided.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": image_paths,
                "fps": 1.0,
            },
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
# 推理部分
# -------------------------------
# 迁移输入到正确的设备
inputs = {key: value.to(model.device) for key, value in inputs.items()}

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

print("📝 推理输出：")
for text in output_text:
    print(text)
