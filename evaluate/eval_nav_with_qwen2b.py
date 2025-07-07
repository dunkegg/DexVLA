import os
import json
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from tqdm import tqdm

instruction = """
You are navigating along a road in a residential area. Your goal is to {Task}.

You are provided with a sequence of images represent your recent movement trajectory.
The last image represents your current view.

Your output must include two parts:
- Reasoning: Clearly explain which objects exist, their location relative to the centerline, and whether they block forward movement.
- Action: Provide the next action (e.g., 'move forward', 'move slightly left and forward', or 'go around the obstacle to the right and continue').

**Important:**  
- You must not assume the road is clear unless you explicitly confirm that no objects overlap the centerline.
- Visual perception errors must be minimized by strictly basing reasoning only on the visual observations provided.


"""

class NavigationEvaluator:
    def __init__(self, model_name_or_path, prompt_template, max_image_num=2):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )
        self.prompt_template = prompt_template

    def build_prompt(self, images, question):
        question = instruction.format(Task=question)
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": self.prompt_template.format(Question=question)}
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def eval(self, samples, batch_size=4):
        results = []
        batch_prompts = []

        for idx, sample in tqdm(enumerate(samples), total=len(samples), desc="Evaluating"):
            images = [Image.open(img_path) for img_path in sample['image_paths']]
            vllm_prompt = self.build_prompt(images, sample['question'])
            batch_prompts.append({"prompt": vllm_prompt, "multi_modal_data": {"image": images}})

            if len(batch_prompts) == batch_size or idx == len(samples) - 1:
                outputs = self.model.generate(batch_prompts, sampling_params=self.sampling_params, use_tqdm=False)
                for output in outputs:
                    results.append(output.outputs[0].text)
                batch_prompts = []

        return results

if __name__ == "__main__":
    # 直接硬编码路径和参数
    model_name_or_path = "checkpoints/Reason-RFT-Spatial-Transformation-Qwen2-VL-2B"
    image_dir = "unpacked_images"
    question_file = "sample/question.json"
    output_file = "output.json"
    batch_size = 1

    # 自定义轨迹推理Prompt模板
    my_navigation_prompt = """{Question} Output the thinking process in <think> </think> and final trajectory plan in <answer> </answer> tags."""

    evaluator = NavigationEvaluator(
        model_name_or_path=model_name_or_path,
        prompt_template=my_navigation_prompt
    )

    # 读取问题
    with open(question_file, 'r') as f:
        raw_samples = json.load(f)

    samples = []
    for group in raw_samples:
        question = group['question']
        image_paths = [os.path.join(image_dir, img) for img in group['images']]
        samples.append({"question": question, "image_paths": image_paths})

    # 推理
    preds = evaluator.eval(samples, batch_size=batch_size)
    print(preds)
    # 保存结果
    output = []
    for sample, pred in zip(raw_samples, preds):
        output.append({"images": sample["images"], "question": sample["question"], "prediction": pred})

    output_dir = os.path.dirname(output_file)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Saved predictions to {output_file}")
