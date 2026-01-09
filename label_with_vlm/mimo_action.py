
import time
import re
import base64
from PIL import Image
from io import BytesIO


from .rxr_utils import (
    encode_image,
    batch_image_indices,
    group_indices_by_string,
)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/mnt/pfs/3zpd5q/code/mimo/DexVLA/checkpoints/MiMo-Embodied-7B" 
HEIGHT = 480
WIDTH = 640
PROMPT_TEMPLATE = (
    f""""You are an autonomous navigation assistant.
    Your task is to <instruction>,
    Where should you go next to stay on track? 
    Please output the next waypoint's coordinate in the image, remember the waypoint should on the floor. The height and width of camera sight is {HEIGHT} and {WIDTH}. so the coordinate must be integers like (h,w), <0=h<={HEIGHT},  <0=w<={WIDTH}. 
    Please output STOP when you have successfully completed the task.
    You should only output like this: "(h,w)" OR "STOP"  
    """
)

PROMPT_TEMPLATE = (
    f""""You are an autonomous navigation assistant.
    Your task is to <instruction>,
    Where should you go next to stay on track? 
    Please output the next waypoint's coordinate in the image, remember the waypoint should on the floor. The height and width of camera sight is {HEIGHT} and {WIDTH}. so the coordinate must be integers like (h,w), <0=h<={HEIGHT},  <0=w<={WIDTH}. 
    Please output STOP when you have successfully completed the task.
    Do not include <think>, explanations, or any extra text.
    Only output like this: "(h,w)" OR "STOP"  
    """
)

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import torch
# class MimoAction:
#     def __init__(self, window_size=10):
#         self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             MODEL_PATH,
#             torch_dtype="auto",
#             device_map="auto"
#         )
#         self.model.eval()

#     def extract_answer(self, text: str) -> str:
#         """
#         提取最后一个 <think> 标签之后的内容
#         """
#         parts = text.split("<think>")
#         # 取最后一个 <think> 后的内容
#         last_think = parts[-1]
#         # 如果有 </think>，只取 </think> 之后的内容
#         if "</think>" in last_think:
#             answer = last_think.split("</think>")[-1]
#         else:
#             answer = last_think
#         return answer.strip()
    
#     def get_action(self, instruction,cur_img_path, history_img_path):
#         history_images = [Image.open(p).convert("RGB") for p in history_img_path]
#         cur_img = Image.open(cur_img_path).convert("RGB")
#         history_image_base64 = [(encode_image(image)) for image in history_images]
#         cur_img_base64 = encode_image(cur_img)

#         prompt = PROMPT_TEMPLATE.replace("<instruction>", instruction)
#         message = {
#             "role": "user",
#             "content": [
#                 # {"type": "text", "text": f'Your task is {prompt}.  You are given a sequence of historical visual observations in temporal order:'},
#                 {"type": "text", "text": f'Your task is {prompt}.'},
#                 # {
#                 #     "type": "video",
#                 #     "video": history_image_base64,
#                 #     "fps": 1.0,
#                 # },
#                 # {"type": "text", "text": 'This is your current observation: '},
#                 # {"type": "image", "image": cur_img_base64}
#             ],
#         }
#         # for base64_image in history_image_base64:
#         #     message["content"].append(
#         #         {
#         #             "type": "image_url",
#         #             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#         #         }
#         #     )
#         message["content"].append({"type": "text", "text": 'This is your current observation: '})
#         message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_img_base64}"}})
        
#         start_time = time.time()
#         text_input = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(message)
#         inputs = self.processor(
#             text=[text_input],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt"
#         ).to(DEVICE)
#         # ========== 推理 ==========
#         print("[INFO] Running model inference...")
#         with torch.no_grad():
#             generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
#         print("all time", time.time() - start_time)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         generated_text = output_text[0]

#         answer = self.extract_answer(generated_text)
#         print(f"[INFO] Generated text:\n{generated_text}")
#         print(f"[INFO] +++++++++++++++++++++++++++++++++++++++")
#         print(f"[INFO] answer:\n{answer}")

#         print("reference_time", time.time() - start_time)
#         # ========== 坐标解析 ==========
#         match = re.search(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", answer)

#         if match:
#             h = int(match.group(1))
#             w = int(match.group(2))
#             coords = (h, w)
#             print("coords", coords)
#             return coords, answer
#         else:
#             if "STOP" in answer:

#                 return None, "STOP"
        

from openai import OpenAI
class MimoActionClient:
    def __init__(self, window_size=10):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.client = client

    def extract_answer(self, text: str) -> str:
        """
        提取最后一个 <think> 标签之后的内容
        """
        parts = text.split("<think>")
        # 取最后一个 <think> 后的内容
        last_think = parts[-1]
        # 如果有 </think>，只取 </think> 之后的内容
        if "</think>" in last_think:
            answer = last_think.split("</think>")[-1]
        else:
            answer = last_think
        return answer.strip()
    
    def get_action(self, instruction,cur_img_path, history_img_path):
        history_images = [Image.open(p).convert("RGB") for p in history_img_path]
        cur_img = Image.open(cur_img_path).convert("RGB")
        history_image_base64 = [(encode_image(image)) for image in history_images]
        cur_img_base64 = encode_image(cur_img)

        prompt = PROMPT_TEMPLATE.replace("<instruction>", instruction)
        message = {
            "role": "user",
            "content": [
                # {"type": "text", "text": f'Your task is {prompt}.  You are given a sequence of historical visual observations in temporal order:'},
                {"type": "text", "text": f'Your task is {prompt}.'},
                # {
                #     "type": "video",
                #     "video": history_image_base64,
                #     "fps": 1.0,
                # },
                # {"type": "text", "text": 'This is your current observation: '},
                # {"type": "image", "image": cur_img_base64}
            ],
        }
        # for base64_image in history_image_base64:
        #     message["content"].append(
        #         {
        #             "type": "image_url",
        #             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        #         }
        #     )
        message["content"].append({"type": "text", "text": 'This is your current observation: '})
        message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_img_base64}"}})
        
        start_time = time.time()

        # 调用 Qwen 模型
        chat_response = self.client.chat.completions.create(
            model="Mimo", messages=[message],  max_tokens=2048, temperature=0
        )
        # 提取输出文本
        generated_text = chat_response.choices[0].message.content.strip()
        answer = self.extract_answer(generated_text)
        print(f"[INFO] Generated text:\n{generated_text}")
        print(f"[INFO] +++++++++++++++++++++++++++++++++++++++")
        print(f"[INFO] answer:\n{answer}")

        print("reference_time", time.time() - start_time)
        # ========== 坐标解析 ==========
        match = re.search(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", answer)

        if match:
            h = int(match.group(1))
            w = int(match.group(2))
            coords = (h, w)
            print("coords", coords)
            return coords, answer
        else:
            if "STOP" in answer:

                return None, "STOP"
        

