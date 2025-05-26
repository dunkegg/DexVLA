# prompt_generator.py

import json
import re
def generate_prompt(instruction, window_size, k, horizon, historical_commands):

    prompt = f"""
    You are a robot walking in the room. This is your global instruction: "{instruction}". 

    The images provided represent your observations along your journey:
    - The first image are your current observation.
    - The following {window_size-1} images are frames showing your future movement.

    Based on these images, please:
    1. Reason: Identify where you are in the task based on the current observation (first image) and future context (following images).
    2. Provide a conclusion: What is the current state you are in, and what will be your next step based on the global instruction.
    3. In order to let you know which step you're on, I will now tell you that the current observation is at step {k+1}/{horizon} in the entire process, and your future states are at {k+2}/{horizon} ~ {k+window_size}/{horizon}.
    4. Your historical commands are: {historical_commands}. Compare them with the global instruction, and the commands that have already appeared indicate that you have completed that part of the task.

    Also:
    Please generate a short action command for each state:
    - For the current state, provide a short command that summarizes current state with verbs.
    - For the future state, provide a short command that summarizes current state with verbs.

    Output the reasoning and conclusion in the following format:
    {{
        "reason": "Your reasoning progress",
        "current_state": "Your current state description",
        "future_state": "Your next step description",
        "current_command": "Your short command for the current state",
        "future_command": "Your short command for the future state"
    }}
    """

    return prompt

def describe_prompt():
    prompt = f"""
        You are a robot executing a VLN (Vision-and-Language Navigation) task.

        The images above represent a segment of your trajectory.

        The **first image** is your **current observation**, and the rest of the images show the **future frames** that follow it.

        Based on this image sequence, please describe what action or sub-task phase the robot initiated **after** the current observation, inferred from the future trajectory.

        The description should be a phase starting with a **verb**.

        Output your reasoning and conclusion in the following format:
        {{
            "reasoning": "your reasoning progress"
            "conclusion": "Your current sub-task phase starting with a verb"
        }}
        """
    return prompt



def action_prompt(instruction, horizon):
    max_actions = max(1, horizon // 5)
    prompt = f"""
    The images represent a robot executing a VLN task: {instruction}.

    You need to tell me for each images, what action should robot do in control level. For example: go forward, turn left, turn right, stop etc .

    Your VLN task is a **global** description of your actions. Use it as a reference only, and give the **local** actions. 
    
    Use actions like:
    - go forward
    - turn left
    - turn right
    - turn around
    - stop

    And modify them using only the following **adverbs to indicate degree**:
    - slightly
    - moderately
    - sharply
    - slowly
    - quickly

    **Important constraints**:
    1. **Do NOT invent new verbs or adverbs.** Stick to the exact words above. For example, don't use "move ahead", "advance", "briskly", "gently", or similar synonyms.

    Output the action and index of images in the following format:
    {{
        "reason": "your reasoning",
        "index of images": "action 1",
        "index of images": "action 2",
        ...
    }}
    Remember there is {horizon} images in total. If you are not sure, you can jump some images.

    For example:  "0-7": "*******" , "8": ********

    **Instructions:**
    1. Ensure the actions are **temporally coherent!!!** — the action assigned to each image should consider its neighboring frames (both before and after).
    3. Each action segment must cover **no more than 10 consecutive images**. If an action spans more than 10 images, break it into chunks like `"0-9"`, `"10-18"` with the same label.
    4. You must label **every image**.
    5. Output the result as a JSON dictionary in this format:

    """
    return prompt

# def action_prompt(instruction, horizon):
#     max_actions = max(1, horizon // 5)
#     prompt = f"""
#     The images represent a robot executing a VLN task: {instruction}.

#     You need to tell me for each images, what action should robot do in control level. For example: go forward, turn left, turn right, stop etc.
#     Output the action and index of images in the following format:
#     {{
#         "index of images": "action 1",
#         "index of images": "action 2",
#         ...
#     }}
#     Remember there is {horizon} images in total. If you are not sure, you can jump some images.

#     For example:  "0-7": "*******" , "8": ********

#     **Instructions:**
#     1. Ensure the actions are **temporally coherent** — the action assigned to each image should consider its neighboring frames (both before and after).
#     2. You can split the trajectory into meaningful action segments, but you should use **at most {max_actions} distinct action types** across the whole sequence.
#     3. Each action segment must cover **no more than 10 consecutive images**. If an action spans more than 10 images, break it into chunks like `"0-9"`, `"10-18"` with the same label.
#     4. You must label **every image**.
#     5. Output the result as a JSON dictionary in this format:

#     """
#     return prompt


# def action_prompt(instruction, horizon):
#     max_actions = max(1, horizon // 5)
#     prompt = f"""
#     The images represent a robot's actions.

#     You need to tell me for each images, what action should robot do in control level. For example: go forward, turn left, turn right, stop etc.
#     Output the action and index of images in the following format:
#     {{
#         "index of images": "action 1",
#         "index of images": "action 2",
#         ...
#     }}
#     Remember there is {horizon} images in total. If you are not sure, you can jump some images.

#     For example:  "0-7": "*******" , "8": ********

#     **Instructions:**
#     1. Ensure the actions are **temporally coherent** — the action assigned to each image should consider its neighboring frames (both before and after).
#     2. You can split the trajectory into meaningful action segments, but you should use **at most {max_actions} distinct action types** across the whole sequence.
#     3. Each action segment must cover **no more than 10 consecutive images**. If an action spans more than 10 images, break it into chunks like `"0-9"`, `"10-18"` with the same label.
#     4. You must label **every image**.
#     5. Output the result as a JSON dictionary in this format:

#     """
#     return prompt


def split_prompt(instruction, horizon):

    prompt = f"""
    The images represent a robot executing a VLN task: {instruction}.

    You need to tell me for each sub-task start with a verb, which images belongs to this sub-task.
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
    For example:  "0-8": "***********"

    Make sure that no single image group (value of the key) contains more than 10 consecutive images.
    If you believe a sub-task spans more than 10 images, you can split it into smaller chunks (e.g., [0-9], [10-19]) and assign all of them to the same sub-task description.

    Every sub-task should be included!
    """
    return prompt


    # Make sure that no single image group (value of the key) contains more than 10 consecutive images.
    # If you believe a sub-task spans more than 10 images, you can split it into smaller chunks (e.g., [0-9], [10-19]) and assign all of them to the same sub-task description.


def split_instruction_prompt(instruction, horizon):
    prompt = f"""
You are a robot executing the following Vision-and-Language Navigation (VLN) instruction:

"{instruction}"

You are given {horizon} consecutive images that visualize your execution of this instruction.

Your tasks:
1. **Split the instruction into two consecutive sub-tasks**, each starting with a verb and **directly copied** from the instruction. These two sub-tasks must be **non-overlapping**, and together they should **reconstruct the original instruction exactly**.
2. **Assign each sub-task to a portion of the image sequence**. The assignment does **not need to be an exact split**, but should reasonably reflect the visual transition between the two sub-tasks.

Important:
- Do **not create new phrasing**. The sub-task descriptions must be **copied verbatim** from the instruction.
- Try to make the division of image indices reflect the semantic change between the sub-tasks, not just the midpoint.
- If necessary, you can adjust the index ranges (e.g., "0-8" and "9-23"), but make sure they cover all images.
- Each sub-task should have at least one verb.
Output format:
{{
    "image index range": "first sub-task (copied from instruction)",
    "image index range": "second sub-task (copied from instruction)"
}}
"""
    return prompt

def extract_ranges_and_descriptions(json_data):
    """
    从模型生成的 JSON 中提取每段图像索引范围和对应的描述。

    Args:
        json_data (dict or str): 类似 {"0-10": "Walk to door", "11-23": "Enter the room"}

    Returns:
        List[Tuple[Tuple[int, int], str]]: 每个元素是 ((start_idx, end_idx), description)
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    results = []
    for indices_range, description in json_data.items():
        # 清理前后括号
        indices_range = re.sub(r'^[\[\{](.*?)[\]\}]$', r'\1', indices_range)

        if '-' in indices_range:
            start_idx, end_idx = map(int, indices_range.split('-'))
        else:
            start_idx = end_idx = int(indices_range)

        results.append(((start_idx, end_idx), description))

    return results

def fill_descriptions(horizon, json_data):
    """
    根据给定的 JSON 数据填充描述信息。

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
        indices_range = re.sub(r'^[\[\{](.*?)[\]\}]$', r'\1', indices_range)
        # 解析索引范围
        if 'reason' in indices_range:
            continue
        if '-' in indices_range:  # 处理 "start-end" 形式的范围
            start_idx, end_idx = map(int, indices_range.split('-'))
            for i in range(start_idx, end_idx + 1):
                if i>=0 and i < horizon:
                    descriptions[i] = description 
        else:  # 处理单个索引，如 "34"
            idx = int(indices_range)
            if idx == horizon:
                idx = idx - 1
            descriptions[idx] = description

    return descriptions

def extract_reasoning_and_conclusion(json_data):
    """
    提取 JSON 数据中的 'reasoning' 和 'conclusion' 字段内容。

    Args:
        json_data (str or dict): JSON 数据（字符串或字典）。

    Returns:
        tuple: ('reasoning' 字段的内容, 'conclusion' 字段的内容)
    """
    # 如果 json_data 是字符串类型，先解析为字典
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    # 提取 'reasoning' 和 'conclusion'
    reasoning = json_data.get('reasoning', '')
    conclusion = json_data.get('conclusion', '')

    return reasoning, conclusion