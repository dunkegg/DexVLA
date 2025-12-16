import h5py
import os
import numpy as np
from rxr_agent_label import QwenLabeler
import gc
import logging


def qwen_lable_images(h5_file_path, labeller):
    """
    对同一批图片，依次生成动作标注和状态标注，并保存到 HDF5 文件中。
    """
    print(f"\n🟢 开始处理文件: {h5_file_path}")
    with h5py.File(h5_file_path, "a") as f:
        # -----------------------------
        # Step 1. 读取原始图片和文字说明
        # -----------------------------
        images = f["obs"]["color_0_0"][:]
        instructions = f["instructions"][:]
        instructions = [i.decode("utf-8") for i in instructions]
        # print("instruction:",instructions)
        actions = f["annotations_actions0"][:]
        actions = [a.decode("utf-8") for a in actions]
        # print("instruction:",actions)
        print(f"📂 读取到 {len(images)} 张图片")
        # -----------------------------
        # Step 2. 状态标注
        # -----------------------------
        status_dataset_name = "annotations_status0"
        status_annotations = None
        need_generate_status = True

        if status_dataset_name in f:
            annotations = f[status_dataset_name][()]
            # 如果第一条不是空字符串，则跳过
            if annotations[0].decode("utf-8") != "":
                print("⏭️ 状态标注已存在，跳过生成。")
                need_generate_status = False

        if need_generate_status:
            print("🤖 开始进行状态标注...")
            status_result = try_label_images(
                labeller.label_images_status,
                images=images,
                instructions=instructions,
                actions=actions,
            )
            if status_result is not None:
                status_annotations, _ = status_result
                save_annotations(
                    f, dataset_name=status_dataset_name, annotations=status_annotations
                )
        print(f"✅ {h5_file_path} 状态标注全部完成。\n")
        f.flush()
        del f
        gc.collect()


def try_label_images(label_func, images, instructions, *args, max_retry=5, **kwargs):
    """
    通用标注器：支持额外参数透传，并保留重试机制。
    """
    for attempt in range(1, max_retry + 1):
        try:
            print(f"🤖 第 {attempt} 次尝试调用 QwenLabeler...")
            result = label_func(images, instructions, *args, **kwargs)
            return result
        except Exception as e:
            print(f"⚠️ 第 {attempt} 次调用出错: {e}")

    print(f"❌ 标注失败（已重试 {max_retry} 次）")
    return None


def save_annotations(h5_file, dataset_name, annotations):
    """
    将标注结果写入 HDF5。
    """
    if dataset_name in h5_file:
        del h5_file[dataset_name]

    cleaned = [s if s is not None else "null" for s in annotations]
    h5_file.create_dataset(
        dataset_name, data=np.array(cleaned, dtype=h5py.string_dtype(encoding="utf-8"))
    )
    print(f"💾 已保存数据集: {dataset_name} ({len(annotations)} 条)")


"""单个hdf5文件"""
# if __name__ == "__main__":
#     logging.basicConfig(filename="debug.log", level=logging.INFO, filemode="w")
#     labeller = QwenLabeler()

#     # 修改此路径为单个文件或目录：
#     h5_file_path = "rxr2_smooth/episode_52.hdf5"
#     qwen_lable_images(h5_file_path, labeller)
"""整个文件夹下的hdf5文件"""
if __name__ == "__main__":
    logging.basicConfig(filename="debug.log", level=logging.INFO, filemode="w")
    labeller = QwenLabeler()

    rxr_dir = "data/raw_data/rxr_smooth/"
    for fname in sorted(os.listdir(rxr_dir)):
        if fname.endswith(".hdf5"):
            path = os.path.join(rxr_dir, fname)
            print(f"==============================")
            print(f"🚀 开始处理文件：{path}")
            print(f"==============================")
            qwen_lable_images(path, labeller)

    print("\n🎉 全部 HDF5 文件标注完成！")
