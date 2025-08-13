import os
import random
import shutil

def random_copy_files(src_folder, dst_folder, num_files):
    # 确保目标文件夹存在
    os.makedirs(dst_folder, exist_ok=True)

    # 获取源文件夹中的所有文件（不包括子文件夹）
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # 如果请求的数量超过文件总数，就复制全部
    if num_files > len(files):
        print(f"⚠️ 你请求复制 {num_files} 个文件，但源文件夹中只有 {len(files)} 个文件，将全部复制。")
        num_files = len(files)

    # 随机选择文件
    selected_files = random.sample(files, num_files)

    # 复制文件
    for filename in selected_files:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.copy2(src_path, dst_path)  # copy2 保留修改时间等元数据
        print(f"Copied: {filename}")

if __name__ == "__main__":
    # ✅ 修改为你自己的路径和数量
    source_folder = r"data/split_data/single_follow"
    destination_folder = r"data/split_data/from_single_follow"
    number_of_files_to_copy = 17755

    random_copy_files(source_folder, destination_folder, number_of_files_to_copy)
