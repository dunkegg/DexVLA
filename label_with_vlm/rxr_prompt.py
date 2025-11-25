import json
import re

def status_begin_prompt(sub_task, main_start, main_end, horizon, historical_commands):
    '''
    生成begin batch 的 prompt
    '''
    begin_prompt = f"""
        You are a robot starting a new visual navigation task. 
        This is your global instruction: "{sub_task}".

        You are given 15 sequential images:
        - The first 10 images represent the range you must describe.
        - The last 5 images serve only as future context to help you reason about continuity and direction.

        Based on these images, please:
        1. Reason: Identify your starting location and overall direction of movement based on the initial 15 images.
        2. Provide a conclusion: Describe the current state and what direction the upcoming batch of images is heading toward.

        You only need to generate status summaries for the first **10** states (images 0-9). 
        Do NOT generate any status for the last 5 context images.

        Please output in **strict JSON format** (no explanations, no markdown):
        {{
            "reasoning": "Brief reasoning about your start location and movement direction",
            "0-9": "Summary of the first 10 states using concise action verbs"
        }}
    """
    return begin_prompt


def status_prompt(sub_task, main_start, main_end, horizon, historical_commands):
    '''
    生成mid batch 的 prompt
    '''
    window_size = main_end - main_start + 1
    hist_text = ", ".join(map(str, historical_commands)) if historical_commands else "none"

    mid_prompt = f"""
        You are a robot navigating in a room. Your global instruction is: "{sub_task}".

        You are provided with a sequence of images which includes:
        - Several **context images** (previous and future frames, used ONLY for understanding temporal consistency)
        - A **main window of {window_size} images** that MUST be labeled.

        IMPORTANT:
        - The model input contains context + main window.
        - You MUST label ONLY the {window_size} images inside the main window.
        - DO NOT generate labels for context frames.

        The main window corresponds to the following global frame indices:
        [{main_start} .. {main_end}]
        These {window_size} images represent your current and upcoming observations.
        Index {main_start} is the current state, and {main_start+1}..{main_end} are the future states.

        Additional information:
        - Total trajectory length = {horizon}.
        - Historical commands: {hist_text}. Compare them with the global instruction to maintain consistency.

        Your task:
        1. **Reasoning** — Describe briefly how you interpret the robot's current progress in the task, using both context and main window frames.
        2. **Label {window_size} states** — For each of the {window_size} main window frames, Provide a short verb-based action/state summary.
        3. Ensure **strict temporal coherence**: the sequence of commands should reflect smooth evolution across frames.
        4. Do NOT produce separate sentences for each image.

        Please output in **strict JSON format** (NO markdown, NO extra sentences):
        {{
            "reasoning": "Briefly describe your reasoning progress",
            "{main_start}-{min(main_start+4, main_end)}": "brief state summary",
            "{main_start+5}-{main_end}": "brief state summary"
        }}
    """
    return mid_prompt


def status_end_prompt(sub_task, main_start, main_end, horizon, historical_commands):
    """
    生成end batch的prompt
    """

    hist_text = ", ".join(map(str, historical_commands)) if historical_commands else "none"
    end_prompt = f"""
        You are approaching the final destination of a visual navigation task.
        This is your global instruction: "{sub_task}".
        Historical commands: {hist_text}.

        You are given a sequence of images:
        - The first 5 images are past context (previous states).
        - The remaining images represent your final steps toward the goal (images {main_start}-{main_end}).

        Based on these images, please:
        1. Reason: Explain how the final images indicate that you are close to the goal.
           Focus on visible objects, landmarks, or room characteristics.
        2. Provide a conclusion: Describe the final navigation steps and confirm the arrival or near-arrival state.
        3. Only describe and generate actions for the **last segment** (images {main_start}-{main_end}).Provide a short verb-based action/state summary.

        Please output in **strict JSON format** (no explanations, no markdown):
        {{
            "reasoning": "Reasoning focusing on final goal confirmation using distinctive objects or landmarks",
            "{main_start}-{main_end}": "Summary of final navigation actions that lead to the endpoint"
        }}
    """
    return end_prompt


def fill_descriptions(horizon, json_data, global_offset=0):
    """
    把上一步提取的描述填充回所有帧

    Args:
        image_list (list): 包含图片的列表，长度用于生成描述列表。
        json_data (dict): 包含图片索引范围和描述的字典。

    Returns:
        list: 长度与 image_list 相同的描述列表。
    """
    # 初始化一个与 image_list 长度一致的描述列表，默认值为 None 或空字符串
    descriptions = [None] * horizon
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    # 遍历 JSON 数据，填充描述
    for indices_range, description in json_data.items():
        indices_range = re.sub(r"^[\[\{](.*?)[\]\}]$", r"\1", indices_range)
        # 解析索引范围
        if "reason" in indices_range:
            continue
        if "-" in indices_range:  # 处理 "start-end" 形式的范围
            start_idx, end_idx = map(int, indices_range.split("-"))
            for i in range(start_idx, end_idx + 1):
                local_i = i - global_offset
                if 0 <= local_i < horizon:
                    descriptions[local_i] = description
        else:
            idx = int(indices_range)
            local_i = idx - global_offset
            if 0 <= local_i < horizon:
                descriptions[local_i] = description

    return descriptions


def extract_reasoning_and_conclusion(json_data):
    """
    从推理型 JSON 提取 reasoning 与 conclusion

    Args:
        json_data (str or dict): JSON 数据（字符串或字典）。

    Returns:
        tuple: ('reasoning' 字段的内容, 'conclusion' 字段的内容)
    """
    # 如果 json_data 是字符串类型，先解析为字典
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    # 提取 'reasoning' 和 'conclusion'
    reasoning = json_data.get("reasoning", "")
    conclusion = json_data.get("conclusion", "")

    return reasoning, conclusion
