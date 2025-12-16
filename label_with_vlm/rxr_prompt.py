import json
import re


def status_prompt(main_start, main_end, actions, global_offset):
    """
    生成带有动作参考的 mid-batch 状态推断 prompt
    """
    window_size = main_end - main_start + 1
    local_start = main_start - global_offset
    local_end = main_end - global_offset
    action_slice = actions[local_start : local_end + 1]
    action_slice_str = ", ".join(action_slice)
    # print("action_slice_str:", action_slice_str)
    mid_prompt = f"""
        You are a robot-state labeler. For each frame index, you are given one low-level action.
        You MUST convert each action into a short sentence using ONLY the allowed templates.

        ----------------------------------------------------
        ACTION → TEMPLATE RULES (STRICT):

        1) For action = "go_forward":
        MUST output:
        "Move forward through the <area>."

        2) For actions = "turn_left_slightly" / "turn_right_slightly":
        MUST output one of:
        "Turn slightly left toward the <area>."
        "Turn slightly right toward the <area>."

        3) For actions = "turn_left" / "turn_right":
        MUST output one of:
        "Turn left at the <area> to enter the <next_area>."
        "Turn right at the <area> to enter the <next_area>."

        4) For action = "approaching_final_point":
        MUST output:
        "Move forward through the <area>."

        ----------------------------------------------------
        STRICT STYLE CONSTRAINTS:
        - You MUST use the templates EXACTLY (same wording).
        - Only <area> or <next_area> may be replaced by a short location phrase.
        - NO additional words, NO reasons, NO compound sentences.
        - ONE sentence per frame ONLY(under 15 words).
        - NO adding "gradually", "slowing", "indicating", etc.
        - You MUST output sentences in the SAME ORDER as actions are given.

        ----------------------------------------------------
        Input:
        - Main window size: {window_size}
        - Actions for frames {main_start}-{main_end}: {action_slice_str}

        ----------------------------------------------------

        Output format (STRICT JSON):
        {{
            "reasoning": "Explain briefly how you interpret the robot's task progress using images + provided actions.",
            "{main_start}": "template-based state summary",
            "{main_start + 1}": "template-based state summary",
            ...
            "{main_end}": "template-based state summary",
        }}
        """
    # print(mid_prompt)
    return mid_prompt


def status_end_prompt(main_start, main_end, actions, global_offset):
    """
    生成end batch的prompt(包含动作参考)
    """
    # 提取整个段的动作（如 ['go_forward', 'turn_left', ...]）

    action_seq = ", ".join(actions)
    # print("final_actions:",action_seq)
    end_prompt = f"""
        You are approaching the final destination of a visual navigation task.

        You are given a sequence of images:
        - The first 5 images are past context (previous states).
        - The remaining images ({main_start}-{main_end}) represent the robot's final steps toward the goal.

        Additional information:
        - A low-level action estimation module has analyzed the robot trajectory.
        - The predicted actions for the final segment are:
          {action_seq}
        - Many of these actions are "approaching_final_point", meaning the robot is slowing down and is visually very close to the goal.

        Your job:
        1. Identify the **nearest explicit target object or location** that the robot is stopping at.
           This must be a close, visually dominant target the robot ends its navigation on.
           Examples: "dining table", "kitchen counter", "cabinet", "sofa", "doorway", etc.

        2. Produce ONE short template-based summary describing the final stopping point.
           The target MUST be explicitly named.
           Format must strictly be:
           - "Stopping at the <explicit target>."
        3. Keep it under 15 words.
        4. Output strict JSON only.

        Output format (STRICT JSON):
        {{
            "reasoning": "Brief reasoning using visible spatial cues and provided actions.",
            "{main_start}-{main_end}": "Stopping at the <explicit target>."
        }}
    """
    return end_prompt


def fill_descriptions(horizon, json_data, global_offset=0):
    """
    将 LLM 输出的逐帧或范围描述填入固定长度的列表。
    支持 key 形式: "100", "100-105", "[100] 状态", "{100}", 等。
    """
    descriptions = [None] * horizon

    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    for key, desc in json_data.items():
        # 跳过 reasoning
        if key.strip().lower() == "reasoning":
            continue

        # 清除前后可能的 [] {} 状态标记
        cleaned = re.sub(r"^[\[\{\(]?", "", key)
        cleaned = re.sub(r"[\]\}\)]?$", "", cleaned)

        # 去掉后缀中文“状态”
        cleaned = cleaned.replace("状态", "").strip()

        # 去掉可能存在的冒号
        cleaned = cleaned.replace(":", "").strip()

        # 解析范围形式 "100-105"
        if "-" in cleaned:
            try:
                start, end = map(int, cleaned.split("-"))
            except:
                continue
            for i in range(start, end + 1):
                local_i = i - global_offset
                if 0 <= local_i < horizon:
                    descriptions[local_i] = desc

        else:
            # 单帧形式 "103"
            try:
                i = int(cleaned)
            except:
                continue
            local_i = i - global_offset
            if 0 <= local_i < horizon:
                descriptions[local_i] = desc

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
