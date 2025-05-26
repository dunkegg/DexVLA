import os
from openai import OpenAI
import json
from label.prompt import generate_prompt, split_prompt, describe_prompt , fill_descriptions , extract_reasoning_and_conclusion, split_instruction_prompt, extract_ranges_and_descriptions, action_prompt
from label.utils import get_images_from_path, encode_image, batch_image_indices, create_short_list, map_annotations_to_long_list

from collections import defaultdict

def group_indices_by_string(strings):
    """
    Args:
        strings (List[str]): 一组字符串，例如 new_annotations
    Returns:
        Dict[str, List[int]]: 键是字符串内容，值是它出现的所有索引
    """
    group_map = defaultdict(list)
    for i, s in enumerate(strings):
        group_map[s].append(i)
    return group_map

class QwenLabeler:
    def __init__(self):
        """
        初始化 ImageLabeler 类。

        Args:
            client (OpenAI): 用于与 OpenAI API 通信的客户端。
            image_path (str): 存放图片的路径。
            prompt_generator (function): 用于生成标注的 prompt 函数。
            image_encoder (function): 用于对图片进行编码的函数。
        """
        # 初始化 OpenAI 客户端
        openai_api_key = "EMPTY"  # 替换为你的 API 密钥
        openai_api_base = "http://localhost:8000/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        self.client = client


    def label_images_action(self, images, sub_tasks):

        groups = group_indices_by_string(sub_tasks)

        encoded_images = []
        for image in images:
            base64_image = encode_image(image) # base64 encoding.
            encoded_images.append(base64_image)
        
        labeled_images = [None] *  len(encoded_images)

        for sub_task, indices in groups.items():
            if sub_task == 'null' or len(indices) == 0:
                continue
            sub_encoded_images = [encoded_images[i] for i in indices]
            prompt = action_prompt(sub_task, len(sub_encoded_images))
            json_output = self.generate_label(sub_encoded_images, prompt)
            sub_images = fill_descriptions(len(sub_encoded_images), json_output)
            index = indices[0]
            for i in indices:
                labeled_images[i] = sub_images[i-index]

        return labeled_images

    def label_images_short(self, images, instruction, type, positions = None, rotations = None):
        encoded_images = []
        for image in images:
            base64_image = encode_image(image) # base64 encoding.
            encoded_images.append(base64_image)
        
        labeled_images = [None] *  len(encoded_images)

        long_list = list(range(len(encoded_images)))
        short_list = create_short_list(long_list, 1)
        short_encoded_images = [encoded_images[i] for i in short_list]
        prompt = split_prompt(instruction, len(short_encoded_images))
        json_output = self.generate_label(short_encoded_images, prompt)
        labeled_images = fill_descriptions(len(short_encoded_images), json_output)
        long_list_with_annotations = map_annotations_to_long_list(short_list, long_list, labeled_images)
        labeled_images = long_list_with_annotations

        return labeled_images

    def label_images_long(self, images, instruction, type, positions = None, rotations = None):
        encoded_images = []
        for image in images:
            base64_image = encode_image(image) # base64 encoding.
            encoded_images.append(base64_image)
        
        labeled_images = [None] *  len(encoded_images)

        prompt = split_instruction_prompt(instruction, len(encoded_images))
        json_output = self.generate_label(encoded_images, prompt)
        split_result = extract_ranges_and_descriptions(json_output)
        for (start_idx, end_idx), sub_instruction in split_result:
            # 取出当前范围内的图片（base64编码）
            image_segment = encoded_images[start_idx:end_idx + 1]
            prompt = split_prompt(sub_instruction, len(image_segment))
            json_output = self.generate_label(image_segment, prompt)
            sub_labeled_images = fill_descriptions(len(image_segment), json_output)

            for i, desc in enumerate(sub_labeled_images):
                labeled_idx = start_idx + i
                # if 0 <= labeled_idx < len(labeled_images):
                labeled_images[labeled_idx] = desc



        return labeled_images

    def generate_label(self, encoded_images, prompt):

        
        message = {
            "role": "user",
            "content": [
            ],
        }


        message["content"].append({"type": "text", "text": prompt})

        for base64_image in encoded_images:
            new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            message["content"].append(new_image)

        chat_response = self.client.chat.completions.create(
            model="Qwen-72B",  # 使用你的模型名称
            messages=[message]
        )
        response_text = chat_response.choices[0].message.content
        cleaned_data = response_text.strip('```json\n').strip('```')
        print("Chat completion output:", cleaned_data)

        return cleaned_data



