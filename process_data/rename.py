import os

def rename_files_in_folder(folder_path, prefix="_file"):
    # 获取该文件夹下所有文件（不包括子文件夹）
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 排序可选：根据文件名排序
    files.sort()

    for idx, filename in enumerate(files, 1):
        # 获取扩展名（如 .txt, .jpg）
        name, ext = os.path.splitext(filename)
        
        # 构建新的文件名
        new_name = f"{name}{prefix}{ext}"
        
        # 构建完整路径
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        
        # 重命名
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    # ⚠️ 修改为你自己的文件夹路径
    folder = r"data/split_data/turning"
    rename_files_in_folder(folder, prefix="_turn")
