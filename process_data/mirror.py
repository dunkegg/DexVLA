import h5py
import numpy as np
from pathlib import Path

def process_action_x_yaw(input_path: Path, output_folder: Path, tag: str = "mirror"):
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{input_path.stem}_{tag}.hdf5"

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        # 复制所有内容
        def recursive_copy(name, obj):
            if isinstance(obj, h5py.Group):
                fout.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                fout.create_dataset(name, data=obj[()], dtype=obj.dtype)

        fin.visititems(recursive_copy)

        # 修改 action 中的 x 和 yaw（第 0 和 2 列）
        if "action" in fout:
            action = fout["action"][:]
            action[:, 0] = -action[:, 0]  # 取负 x
            action[:, 2] = -action[:, 2]  # 取负 yaw
            del fout["action"]
            fout.create_dataset("action", data=action, dtype="float32")
            # fout.create_dataset("tag", data="mirror")
            fout.attrs["tag"] = "mirror"
        else:
            print(f"❌ 'action' not found in {input_path.name}")
            return

    print(f"✅ 处理完成：{output_path.name}")

def process_folder(input_folder: str, output_folder: str, tag="mirror"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if not input_folder.exists():
        print("❌ 输入文件夹不存在")
        return

    h5_files = list(input_folder.glob("*.hdf5"))
    if not h5_files:
        print("📂 没有找到 .hdf5 文件")
        return

    print(f"📂 共找到 {len(h5_files)} 个文件，开始处理...\n")
    for f in h5_files:
        process_action_x_yaw(f, output_folder, tag=tag)

if __name__ == "__main__":
    # 修改路径：输入文件夹 + 输出文件夹
    process_folder(
        input_folder="data/split_data/single_follow",
        output_folder="data/split_data/mirror_sum",
        tag="mirror"
    )
