import os
import json
from datetime import datetime

def create_log_json(log_dir="logs"):
    """生成一个以当前日期时间命名的空 JSON 文件"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"log_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)

    # 初始化一个空的 JSON 数组
    with open(filepath, "w") as f:
        json.dump([], f, indent=4)

    return filepath


def append_log(filepath, index, success, sample_fps, plan_fps, follow_size):
    """向 JSON 文件中追加一条记录"""
    # 读取当前内容
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = []

    # 添加新的记录
    new_entry = {
        "index": index,
        "success": success,
        "sample_fps": sample_fps,
        "plan_fps": plan_fps,
        "follow_size": follow_size
    }
    data.append(new_entry)

    # 保存回文件
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)