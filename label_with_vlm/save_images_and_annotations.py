import h5py
import numpy as np
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

def wrap_text_by_width_en(text, draw, font, max_width):
    """
    全英文文本自动按单词换行。
    - text: 原始文本
    - draw: ImageDraw 对象
    - font: PIL 字体
    - max_width: 最大像素宽度
    返回: list of lines
    """
    words = text.split(" ")
    lines = []
    line = ""

    for word in words:
        test_line = line + (" " if line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            line = test_line
        else:
            if line:  # 先保存当前行
                lines.append(line)
            line = word  # 新行从当前单词开始

    if line:
        lines.append(line)

    return lines


def save_images_with_annotations(h5_path, output_dir):
    # 若目录已存在，清空
    if os.path.exists(output_dir):
        print(f"⚠️ 检测到已存在的输出目录 {output_dir}，正在清空旧文件...")
        # ⚠️ 警告：如果您希望保留旧文件，请注释掉或删除下一行
        shutil.rmtree(output_dir) 
    os.makedirs(output_dir, exist_ok=True)

    # 打开 h5 文件
    try:
        with h5py.File(h5_path, "r") as f:
            images = f["obs"]["color_0_0"][:]  # shape (N, H, W, 4)
            filenames = f["filenames"][:] if "filenames" in f else None

            # --- 关键修改 1: 读取 status_0 和 status_1 ---
            # 假设修复后的结构是 /annotations_status/status_X
            try:
                status_0_annotations = f["annotations_status"]["status_0"][:]
                status_1_annotations = f["annotations_status"]["status_1"][:]
            except KeyError as e:
                print(f"❌ 错误：无法找到键 {e}。请检查 HDF5 文件结构是否已修复！")
                return
            # --- 关键修改 1 结束 ---

            print(f"📂 读取 {len(images)} 张图片。")

            for i, img_data in enumerate(images):
                # 转换为 RGB 图片
                img = Image.fromarray(img_data.astype(np.uint8)).convert("RGB")
                draw = ImageDraw.Draw(img)
                # 假设字体路径正确
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=16)
                except IOError:
                    print("⚠️ 字体文件未找到，使用默认字体。")
                    font = ImageFont.load_default()
                
                max_width = img.width // 2
                y_offset = 10 # 初始垂直偏移量

                # ===================================================
                # 绘制 Status 0
                # ===================================================
                status_0_text = status_0_annotations[i].decode("utf-8") if isinstance(status_0_annotations[i], bytes) else str(status_0_annotations[i])
                
                draw.text((10, y_offset), "Status 0:", fill=(255, 255, 255), font=font)
                y_offset += 20
                
                lines_0 = wrap_text_by_width_en(status_0_text, draw, font, max_width)
                for line in lines_0:
                    draw.text((10, y_offset), line, fill=(255, 0, 0), font=font) # 红色
                    y_offset += 20
                
                # 添加一个分隔，确保 Status 1 不会紧贴着 Status 0
                y_offset += 10 
                
                # ===================================================
                # 绘制 Status 1 (紧接着 Status 0 之后)
                # ===================================================
                status_1_text = status_1_annotations[i].decode("utf-8") if isinstance(status_1_annotations[i], bytes) else str(status_1_annotations[i])
                
                draw.text((10, y_offset), "Status 1:", fill=(255, 255, 255), font=font)
                y_offset += 20
                
                lines_1 = wrap_text_by_width_en(status_1_text, draw, font, max_width)
                for line in lines_1:
                    draw.text((10, y_offset), line, fill=(0, 255, 255), font=font) # 青色/浅蓝色，区分 Status 0
                    y_offset += 20


                # 文件名
                filename = filenames[i].decode("utf-8") if filenames is not None else f"img_{i+1:04d}.jpg"
                save_path = os.path.join(output_dir, filename)
                img.save(save_path)

                # 打印进度
                if (i + 1) % 10 == 0 or i == len(images) - 1:
                    print(f"✅ 已保存 {i+1}/{len(images)} 张图片 -> {save_path}")

        print(f"🎉 文件 {os.path.basename(h5_path)} 的所有图片已保存到：{output_dir}")

    except Exception as e:
        print(f"致命错误：处理文件 {h5_path} 时发生异常：{e}")
        # 打印详细堆栈，方便调试
        import traceback
        traceback.print_exc()

'''单个episode'''
if __name__ == "__main__":
    # 单个 HDF5 文件路径
    h5_path = "data/raw_data/rxr_smooth/episode_0.hdf5"
    
    # 输出根目录
    output_root = "data/raw_data/rxr_smooth/"
    os.makedirs(output_root, exist_ok=True)
    
    # 使用文件名作为输出子目录
    episode_name = os.path.splitext(os.path.basename(h5_path))[0]  # episode_0
    output_dir = os.path.join(output_root, episode_name)

    print(f"\n🚀 开始处理 {h5_path} ...")
    save_images_with_annotations(h5_path, output_dir)

    print("\n🎉🎉🎉 已成功导出该 episode 的图片与标注！")

'''整个文件夹读取'''
# if __name__ == "__main__":
#     rxr_dir = "/wangzejin/code/DexVLA/hyz_test/rxr2"
#     output_root = "/wangzejin/code/DexVLA/hyz_test/extracted_images"

#     os.makedirs(output_root, exist_ok=True)

#     # 遍历所有 episode_x.hdf5
#     files = sorted([f for f in os.listdir(rxr_dir) if f.endswith(".hdf5")])

#     for fname in files:
#         h5_path = os.path.join(rxr_dir, fname)
#         episode_name = os.path.splitext(fname)[0]  # 如 episode_0
#         output_dir = os.path.join(output_root, episode_name)

#         print(f"\n🚀 开始处理 {fname} ...")
#         save_images_with_annotations(h5_path, output_dir)

#     print("\n🎉🎉🎉 全部 episode 已成功导出图片与标注！")
