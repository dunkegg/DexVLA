import os
import shutil

def merge_folders(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # 定义内部函数：复制一个文件夹下所有文件到目标文件夹中
    def copy_all(src_folder):
        for root, dirs, files in os.walk(src_folder):
            # 计算相对路径（保留原始结构）
            rel_path = os.path.relpath(root, src_folder)
            target_path = os.path.join(output_folder, rel_path)
            os.makedirs(target_path, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_path, file)

                # # 如果目标已存在同名文件，重命名（防止覆盖）
                # if os.path.exists(dst_file):
                #     base, ext = os.path.splitext(file)
                #     count = 1
                #     while True:
                #         new_name = f"{base}_copy{count}{ext}"
                #         new_path = os.path.join(target_path, new_name)
                #         if not os.path.exists(new_path):
                #             dst_file = new_path
                #             break
                #         count += 1

                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")

    # 复制两个源文件夹
    copy_all(folder1)
    # copy_all(folder2)

    print(f"\n✅ 合并完成，新文件夹路径：{output_folder}")

if __name__ == "__main__":
    # ✅ 修改为你自己的路径
    folder1 = r"data/split_data/single_follow_mirrored"
    folder2 = r"data/split_data/single_follow"
    output = r"data/split_data/mirror_sum"

    merge_folders(folder1, folder2, output)
