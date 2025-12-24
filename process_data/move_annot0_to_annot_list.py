import h5py
import numpy as np
from pathlib import Path

def move_annotations_status0_to_group(h5_dir):
    h5_dir = Path(h5_dir)
    assert h5_dir.is_dir(), f"{h5_dir} is not a directory"

    for h5_path in h5_dir.glob("*.hdf5"):
        print(f"Processing: {h5_path}")

        with h5py.File(h5_path, "r+") as f:

            # 1. 检查 annotations_status0 是否存在
            if "annotations_status0" not in f:
                print("  - annotations_status0 not found, skip")
                continue

            status0 = f["annotations_status0"][()]

            # 2. 过滤掉 "null"
            cleaned = []
            for x in status0:
                if isinstance(x, bytes):
                    x = x.decode("utf-8")
                if x != "null":
                    cleaned.append(x)
                else:
                    cleaned.append("null")  # 保持长度一致（更安全）

            cleaned = np.array(cleaned, dtype=h5py.string_dtype("utf-8"))

            # 3. 创建 / 获取 group
            grp = f.require_group("annotations_status")

            # 4. 写入 status_2（如果已存在先删除）
            if "status_2" in grp:
                del grp["status_2"]

            grp.create_dataset("status_2", data=cleaned)

            # 5. 删除原来的 annotations_status0
            del f["annotations_status0"]

            print("  ✓ moved to annotations_status/status_2")

    print("All files processed.")

move_annotations_status0_to_group("data/raw_data/rxr_smooth")
