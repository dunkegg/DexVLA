import json
from collections import defaultdict

def group_by_scan(json_path, output_path):
    # 读取原始 JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    scan_dict = defaultdict(list)

    # 遍历每一条 entry
    for item in data:
        scan = item["scan"]
        scan_dict[scan].append(item)

    # 转成普通 dict 才能保存成 JSON
    scan_dict = dict(scan_dict)

    # 保存输出 JSON
    with open(output_path, "w") as f:
        json.dump(scan_dict, f, indent=4)

    print(f"Saved grouped result to {output_path}")

group_by_scan(
    json_path="R2R_data_augmentation_paths_distilled.json",
    output_path="RxR_10000.json"
)