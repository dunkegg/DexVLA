import os
from pathlib import Path

def delete_copy1_hdf5(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 文件夹不存在：{folder_path}")
        return

    deleted_count = 0
    for file in folder.rglob("*copy1.hdf5"):  # 递归查找所有 copy1.hdf5 结尾的文件
        try:
            file.unlink()  # 删除文件
            print(f"🗑️ 已删除：{file}")
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ 删除失败：{file}，原因：{e}")

    print(f"\n✅ 删除完成，共删除 {deleted_count} 个文件")

# 示例用法
if __name__ == "__main__":
    delete_copy1_hdf5("data/split_data/mirror_sum")  # 替换为你的目标文件夹路径
