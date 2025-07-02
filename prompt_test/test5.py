
from openai import OpenAI

import json

from label.prompt import generate_prompt
from label.utils import get_images_from_path, encode_image

# 配置 OpenAI API 和 vLLM API 服务器
openai_api_key = "EMPTY"  # 替换为你的 API 密钥
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)



# 推理函数
def get_task_status(image_dir, window_size, instruction):
    # 获取连续图像窗口
    window_images, horizon,all_images = get_images_from_path(image_dir, window_size)
    

    historical_commands = ""
    for k in [1,5,10,20,25,35,45,55,60]:
        message = {
            "role": "user",
            "content": [
            ],
        }

        print(window_images[k][0])
        for i in range(len(window_images[k])):
            base64_image = encode_image(window_images[k][i]) # base64 encoding.
            # new_image = {"type": "input_image", "image_url":  f"data:image/jpeg;base64,{base64_image}"}
            new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            message["content"].append(new_image)

        prompt = f"""
        You are a robot executing a VLN (Vision-and-Language Navigation) task.

        The images above represent a segment of your trajectory.

        The **first image** is your **current observation**, and the rest of the images show the **future frames** that follow it.

        Based on this image sequence, please describe what action or sub-task phase the robot initiated **after** the current observation, inferred from the future trajectory.

        The description should be a phase starting with a **verb**.

        Output your reasoning and conclusion in the following format:
        {{
            "description": "Your current sub-task phase starting with a verb"
        }}
        """


        message["content"].append({"type": "text", "text": prompt})


        # 调用 vLLM API 进行推理
        chat_response = client.chat.completions.create(
            model="Qwen-72B",  # 使用你的模型名称
            messages=[message]
        )
        response_text = chat_response.choices[0].message.content
        cleaned_data = response_text.strip('```json\n').strip('```')
        print(f"Chat completion output: {k} ", cleaned_data)

# return results

# 设置图像路径目录
image_dir = "data/geodesic_path/01/images"
instruction = "walk past staircase, turn left at dining table and stop in front of desk"
instruction = "Turn around and go to your right passed the living area and the stairs. Walk to the left of the table and turn left. Stop in the room with the book on the table."

windwo_size = 10

# 调用推理函数
get_task_status(image_dir, windwo_size,instruction)

