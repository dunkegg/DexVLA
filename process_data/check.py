import h5py
import numpy as np
import argparse
from pathlib import Path

def print_all_keys(h5file, prefix=""):
    """递归打印 HDF5 文件的结构"""
    for key in h5file:
        item = h5file[key]
        if isinstance(item, h5py.Group):
            print_all_keys(item, prefix + key + "/")
        else:
            print(f"{prefix}{key}  ->  shape: {item.shape}, dtype: {item.dtype}")

def decode_label(label):
    """智能解码：支持 bytes / ndarray / string_ / fallback hex"""
    try:
        if isinstance(label, bytes):
            return label.decode("utf-8")
        elif isinstance(label, np.ndarray):
            return label.tobytes().decode("utf-8")
        elif isinstance(label, np.string_):
            return str(label)
        else:
            return str(label)
    except UnicodeDecodeError:
        # 解码失败时输出十六进制表示
        return "<binary: " + label.tobytes().hex()[:32] + "...>"

def print_structure(h5_path: Path, field: str):
    """打印 HDF5 文件结构和指定字段的内容"""
    with h5py.File(h5_path, "r") as f:
        print("📂 HDF5 文件结构:")
        print("-" * 40)
        print_all_keys(f)
        print("-" * 40)

        if field not in f:
            print(f"❌ 字段 '{field}' 不存在！请确认路径正确（如 'follow_paths/000000/observations/history_images'）")
            return

        dataset = f[field]
        print(f"\n🔍 数据字段 `{field}`:")
        print(f"  Shape: {dataset.shape}")
        print(f"  Dtype: {dataset.dtype}")
        print("  前5项内容：")

        # 读取数据集
        data = dataset[()]
        # 如果是字符串数组（如 |S51），逐个解码
        if dataset.dtype.kind == 'S':  # 固定长度字节字符串
            for i, item in enumerate(data[:]):
                decoded = decode_label(item)
                print(f"  [{i}] {decoded}")
        else:
            # 非字符串数据直接打印
            for i, item in enumerate(data[:]):
                print(f"  [{i}] {item}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", help="HDF5 文件路径")
    parser.add_argument("--field", default="observations/history_images", 
                        help="要预览的字段路径")
    args = parser.parse_args()

    print_structure(Path(args.h5_path), args.field)