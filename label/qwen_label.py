import os
from openai import OpenAI
import json
from label.prompt import generate_prompt, split_prompt, describe_prompt , fill_descriptions , extract_reasoning_and_conclusion
from label.utils import get_images_from_path, encode_image, batch_image_indices, create_short_list, map_annotations_to_long_list

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


    def label_images(self, images, instruction, type):
        encoded_images = []
        for image in images:
            base64_image = encode_image(image) # base64 encoding.
            encoded_images.append(base64_image)
        
        labeled_images = [None] *  len(encoded_images)
        if type == "split":
            long_list = list(range(len(encoded_images)))
            short_list = create_short_list(long_list, 2)
            short_encoded_images = [encoded_images[i] for i in short_list]
            prompt = split_prompt(instruction, len(short_encoded_images))
            json_output = self.generate_label(short_encoded_images, prompt)
            labeled_images = fill_descriptions(len(short_encoded_images), json_output)
            long_list_with_annotations = map_annotations_to_long_list(short_list, long_list, labeled_images)
            labeled_images = long_list_with_annotations

        elif type == "describe":
            batches = batch_image_indices(len(encoded_images), batch_size=10)
            prompt = describe_prompt()
            for batch in batches:
                batch_images = [encoded_images[i] for i in batch] 
                json_output = self.generate_label(batch_images, prompt)
                reasoning, conclusion = extract_reasoning_and_conclusion(json_output)
                for i in batch:
                    labeled_images[i] = conclusion   




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



