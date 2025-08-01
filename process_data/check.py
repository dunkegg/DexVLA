import h5py
import argparse
import os

def inspect_hdf5_group(name, obj):
    """用于打印 HDF5 文件中每个对象的基本信息"""
    print(f"{'='*50}")
    print(f"Name: {name}")
    if isinstance(obj, h5py.Group):
        print("Type: Group")
    elif isinstance(obj, h5py.Dataset):
        print("Type: Dataset")
        print(f"Shape: {obj.shape}")
        print(f"Dtype: {obj.dtype}")
    print(f"{'='*50}")

def inspect_hdf5_file(file_path):
    """递归打印 HDF5 文件内容结构"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Opening file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        print("Inspecting file structure...\n")
        f.visititems(inspect_hdf5_group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .hdf5 file contents")
    parser.add_argument("file", type=str, help="Path to the .hdf5 file")
    args = parser.parse_args()
    inspect_hdf5_file(args.file)
