import os
from openai import OpenAI
import json
from collections import defaultdict
from rxr_prompt import (
    status_begin_prompt,
    status_prompt,
    status_end_prompt,
    fill_descriptions,
    extract_reasoning_and_conclusion,
)
from rxr_utils import (
    encode_image,
    batch_image_indices,
    group_indices_by_string,
)


class QwenLabeler:
    def __init__(self):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        # client = OpenAI(
        #     api_key="EMPTY",
        #     base_url="http://localhost:8000/v1",
        #     http_client=openai.OpenAIHTTPClient(proxy=None)
        # )
        self.client = client

    # -------------------------------------------------------------
    # 状态推理
    # -------------------------------------------------------------
    def label_images_status(
        self,
        images,
        instruction,
        window_size=10,
        type=type,
        positions=None,
        rotations=None,
    ):
        groups = group_indices_by_string(instruction)
        encoded_images = [encode_image(img) for img in images]
        num_images = len(encoded_images)

        labeled_images = [None] * len(images)
        reasoning_texts = [None] * len(images)

        historical_command = None
        all_reasonings = []  # 收集每批次 reasoning

        for sub_task, indices in groups.items():
            if sub_task == "null" or len(indices) == 0:
                continue

            # 根据子任务的 indices 产生窗口 batch
            window_batches = batch_image_indices(
                len(indices), window=window_size, pad=5
            )

            for window in window_batches:
                batch_indices = [indices[i] for i in window["indices"]]
                # 主窗口（10 张）闭区间
                main_start, main_end = window["main_range"]
                main_batch_indices = [indices[i] for i in range(main_start, main_end + 1)]

                # 取 batch 的全部图片 (context + main)
                sub_encoded_images = [encoded_images[i] for i in batch_indices]

                # 选择正确的 prompt
                if window["type"] == "end":
                    prompt_fn = status_end_prompt
                else:
                    prompt_fn = status_prompt

                prompt = prompt_fn(
                    sub_task=sub_task,
                    main_start=main_start,
                    main_end=main_end,
                    horizon=num_images,
                    historical_commands=[historical_command] if historical_command else [],
                    type = type,
                )
                # 生成 JSON 输出
                json_output = self.generate_label(sub_encoded_images, prompt)
                reasoning, _ = extract_reasoning_and_conclusion(json_output)
                all_reasonings.append(reasoning)

                # 解析标签
                sub_labels = fill_descriptions(
                    len(sub_encoded_images),
                    json_output,
                    global_offset=batch_indices[0],
                )

                # 只把主窗口 (main_range) 写入最终标注
                for i, g_idx in enumerate(batch_indices):
                    if g_idx in main_batch_indices:  # 仅标主窗口
                        labeled_images[g_idx] = sub_labels[i]
                        reasoning_texts[g_idx] = reasoning

                        # 调试打印
                        if (g_idx + 1) % 10 == 0:
                            print(f"\n--- 第 {g_idx + 1} 张 ---")
                            print(f"推理描述：{reasoning}")
                            print(f"状态标注：{sub_labels[i]}")

                # 更新历史命令，只基于主窗口最后一张
                if main_batch_indices:
                    last_main = main_batch_indices[-1]
                    idx_in_batch = batch_indices.index(last_main)
                    historical_command = sub_labels[idx_in_batch]

        return labeled_images, reasoning_texts, all_reasonings

    # -------------------------------------------------------------
    # 通用生成函数
    # -------------------------------------------------------------
    def generate_label(self, encoded_images, prompt):
        """
        功能：向 Qwen 模型发送带图像的标注请求,返回模型输出(JSON 字符串)
        Args:
            encoded_images (List[str]): base64 编码后的图片
            prompt (str): 构造好的文本提示词
        Returns:
            cleaned_data (str): 模型返回的纯净 JSON 字符串
        """
        # 构建消息
        message = {"role": "user", "content": []}
        message["content"].append({"type": "text", "text": prompt})
        # 添加图片
        for base64_image in encoded_images:
            message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
        # 调用 Qwen 模型
        chat_response = self.client.chat.completions.create(
            model="Qwen-72B", messages=[message]
        )
        # 提取输出文本
        response_text = chat_response.choices[0].message.content.strip()
        # 尝试清理多余包裹符号（如 ```json ... ```）
        cleaned_data = response_text.replace("```json", "").replace("```", "").strip()
        # print("Chat completion output:", cleaned_data)
        return cleaned_data
