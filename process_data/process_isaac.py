import h5py
import os
import numpy as np
import imageio


def extract_traj1_to_new_h5(src_h5, dst_h5, img_save_dir):
    """
    src_h5: 原始 hdf5 路径
    dst_h5: 输出的新 hdf5 路径
    img_save_dir: 保存图片的文件夹，例如 "output_images/"
    """
    os.makedirs(img_save_dir, exist_ok=True)

    with h5py.File(src_h5, "r") as fin, h5py.File(dst_h5, "w") as fout:
        traj1 = fin["traj_1"]

        # ---------------------------
        # 1) 复制 annotations_status（如果存在）
        # ---------------------------
        if "annotations_status" in traj1:
            ds = traj1["annotations_status"]
            fout.create_dataset(
                "annotations_status/status_1",
                data=ds[()],
                dtype=ds.dtype
            )

        # ---------------------------
        # 2) 复制 instruction（如果存在）
        # ---------------------------
        if "instruction" in traj1:
            ds = traj1["instruction"]
            fout.create_dataset(
                "instruction",
                data=ds[()],
                dtype=ds.dtype
            )

        # ---------------------------
        # 3) 处理 observations/images
        # ---------------------------
        if "observations" in traj1 and "images" in traj1["observations"]:
            images = traj1["observations"]["images"][:]  # (N,H,W,3)
            N = images.shape[0]

            path_list = []  # *** 用于保存路径字符串 ***

            print(f"[INFO] Saving {N} images to: {img_save_dir}")

            for i in range(N):
                img = images[i]
                img_filename = f"{i:05d}.png"
                img_path = os.path.join(img_save_dir, img_filename)

                # 保存图片
                # imageio.imwrite(img_path, img)

                # 路径作为字符串加入列表
                path_list.append(img_path)

            # 将路径保存到新 HDF5
            fout.create_dataset(
                "frames",
                data=np.array(path_list, dtype="S256")  # 字符串 dataset
            )


extract_traj1_to_new_h5(
    src_h5="data/proc_data/test_real/episode_000.hdf5",
    dst_h5="data/proc_data/test_real/isaac_f4_C_episode_000.hdf5",
    img_save_dir="data/frames/isaac_f4_C/"
)

# def main(src_dir: Path, frames_dir: Path, dst_dir: Path, viz_dir: Path|None, history:int):
#     src_dir, frames_dir, dst_dir = map(Path,(src_dir,frames_dir,dst_dir))
#     frames_dir.mkdir(parents=True,exist_ok=True)
#     dst_dir.mkdir(parents=True,exist_ok=True)

#     h5_files=sorted(src_dir.glob("*.hdf5"))
#     if not h5_files:
#         print("‼ 未找到 *.hdf5 文件于",src_dir); return

#     count = 0
#     for f in h5_files:
#         count += process_one(f, frames_dir, dst_dir, viz_dir, history)
#         print(f"Processed {count} cases in total.")

# if __name__=="__main__":
#     ap=argparse.ArgumentParser(description="batch convert episodes")
    
#     args=ap.parse_args()
#     args.src_dir = "data/raw_data/raw_single_follow_data"
#     args.frames_dir ="data/frames/single_follow"
#     args.dst_dir = "data/proc_data/single_follow"


#     args.viz = "results/multi_follow"
#     args.history = 10
#     viz=Path(args.viz) if args.viz else None
#     main(Path(args.src_dir), Path(args.frames_dir), Path(args.dst_dir), viz, args.history)
