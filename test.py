import base64
import os
import glob
from openai import OpenAI
import requests

# 配置 OpenAI API 和 vLLM API 服务器
openai_api_key = "EMPTY"  # 替换为你的 API 密钥
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

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
    
    return images

# 编码图像为 base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 推理函数
def get_task_status(image_dir, instruction):
    # 获取连续图像窗口
    window_images = get_images_from_path(image_dir)
    
    for k in range(len(window_images)):
        if k >=1:
            break
        images = window_images[k]
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
            ],
        }
            
        for i in range(len(images)):
            base64_image = encode_image(images[i]) # base64 encoding.
            # new_image = {"type": "input_image", "image_url":  f"data:image/jpeg;base64,{base64_image}"}
            new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            message["content"].append(new_image)

        # 调用 vLLM API 进行推理
        chat_response = client.chat.completions.create(
            model="Qwen-72B",  # 使用你的模型名称
            messages=[message]
        )
            
        print("Chat completion output:", chat_response.choices[0].message.content)
    
    # return results

# 设置图像路径目录
image_dir = "data/geodesic_path/00/images"

instruction="walk past staircase, turn left at dining table and stop in front of desk"
instruction="what do these pictures show?"
# 调用推理函数
get_task_status(image_dir, instruction)

