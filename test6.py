
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
    
    # all_images = all_images[::2]
    # horizon = len(all_images)
    # prompt = f"""
    # You are given a VLN (Vision-and-Language Navigation) instruction:

    # \"{instruction}\"

    # Your task is to break the sentence into individual sub-instructions, **splitting at each conjunction 'and' that connects action verbs**.

    # - Start each sub-instruction with the verb.
    # - Keep the instruction grammatically correct and meaningful.
    # - Do not split noun phrases or prepositional phrases (e.g. don't split 'living area and the stairs').
    # - Output the result as a list, each item is a sub-instruction.

    # Now split:
    # """

    # message = {
    #     "role": "user",
    #     "content": [
    #         {"type": "text", "text": prompt},
    #     ],
    # }

    # chat_response = client.chat.completions.create(
    #     model="Qwen-72B",  # 使用你的模型名称
    #     messages=[message]
    # )
    # instruction = chat_response.choices[0].message.content
    print("++++++++++++++++++++++++++++++++")
    print(f"split instruction: {instruction}")


    message = {
        "role": "user",
        "content": [
        ],
    }
    prompt = f"""
    The images represent a robot executing a VLN task: {instruction}.

    You need to tell me for each sub-task start with a verd, which images belongs to this sub-task.
    Output the sub-task and index of images in the following format:
    {{
        "index of images": "description of sub-task 1",
        "index of images": "description of sub-task 2",
        ...
    }}
    Remember there is {horizon} images in total. If you are not sure, you can jump some images.
    """

    prompt = f"""
    The images represent a robot executing a VLN task: {instruction}.

    You need to segment these {horizon} images into sub-tasks, where each sub-task starts with a verb from the instruction. 
    For each sub-task, specify the indices of the images that belong to it, using the format:

    {{
        "indices of images": "description of sub-task 1",
        "indices of images": "description of sub-task 2",
        ...
    }}

    Make sure that no single image group (value of the key) contains more than 10 consecutive images.
    If you believe a sub-task spans more than 10 images, you can split it into smaller chunks (e.g., [0-9], [10-19]) and assign all of them to the same sub-task description.

    Every sub-task should be included!
    """

    message["content"].append({"type": "text", "text": prompt})
    for i in range(len(all_images)):
        base64_image = encode_image(all_images[i]) # base64 encoding.
        # new_image = {"type": "input_image", "image_url":  f"data:image/jpeg;base64,{base64_image}"}
        new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        message["content"].append(new_image)



    # 调用 vLLM API 进行推理
    chat_response = client.chat.completions.create(
        model="Qwen-72B",  # 使用你的模型名称
        messages=[message]
    )
    response_text = chat_response.choices[0].message.content
    cleaned_data = response_text.strip('```json\n').strip('```')
    print("Chat completion output:", cleaned_data)

# return results

# 设置图像路径目录
image_dir = "data/geodesic_path/00/images"
instruction = "Turn around. Walk past staircase, turn left at dining table and stop in front of desk"
# instruction = "Turn around. Go to your right passed the living area and the stairs. Walk to the left of the table. Turn left. Stop in the room with the book on the table."

windwo_size = 5

# 调用推理函数
get_task_status(image_dir, windwo_size,instruction)

