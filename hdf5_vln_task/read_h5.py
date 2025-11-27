# import h5py
# import numpy as np

# def print_hdf5_item(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         try:
#             data = obj[()]
#             # 如果是 bytes，需要 decode
#             if isinstance(data, (bytes, np.bytes_)):
#                 data = data.decode()
#             print(f"{name}: shape={obj.shape}, dtype={obj.dtype}, data={data}")
#         except Exception as e:
#             print(f"{name}: [Error reading data] {e}")
#     elif isinstance(obj, h5py.Group):
#         print(f"{name}/ (Group)")

# def read_and_print_hdf5_file(filename):
#     print(f"[INFO] Reading HDF5 file: {filename}")
#     with h5py.File(filename, 'r') as f:
#         f.visititems(print_hdf5_item)
#     print("[INFO] Done.\n")

# # 使用示例
# hdf5_filename  = "/mnt/pfs/3zpd5q/code/zf/raw_data/matterport_data/objnav.h5"
# read_and_print_hdf5_file(hdf5_filename)
#==========================show index h5=======================================================#
import h5py
import numpy as np

def print_hdf5_item(name, obj, target_index=None):
    """
    打印HDF5项目内容，可指定特定索引
    
    Args:
        name: 项目名称
        obj: HDF5对象
        target_index: 目标索引，如果为None则打印所有内容
    """
    if isinstance(obj, h5py.Dataset):
        # 检查是否是指定的索引
        if target_index is not None:
            # 只处理指定索引的轨迹组
            if name.startswith(f"traj_{target_index}/"):
                try:
                    data = obj[()]
                    # 如果是 bytes，需要 decode
                    if isinstance(data, (bytes, np.bytes_)):
                        data = data.decode()
                    print(f"{name}: shape={obj.shape}, dtype={obj.dtype}, data={data}")
                except Exception as e:
                    print(f"{name}: [Error reading data] {e}")
        else:
            # 打印所有内容
            try:
                data = obj[()]
                if isinstance(data, (bytes, np.bytes_)):
                    data = data.decode()
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}, data={data}")
            except Exception as e:
                print(f"{name}: [Error reading data] {e}")
    elif isinstance(obj, h5py.Group):
        if target_index is not None:
            # 只打印指定索引的组
            if name == f"traj_{target_index}":
                print(f"{name}/ (Group)")
        else:
            # 打印所有组
            print(f"{name}/ (Group)")

def read_and_print_hdf5_file(filename, target_index=None):
    """
    读取并打印HDF5文件内容
    
    Args:
        filename: HDF5文件名
        target_index: 目标轨迹索引，如果为None则打印所有内容
    """
    if target_index is not None:
        print(f"[INFO] Reading HDF5 file: {filename}, target index: traj_{target_index}")
    else:
        print(f"[INFO] Reading HDF5 file: {filename} (all content)")
    
    with h5py.File(filename, 'r') as f:
        if target_index is not None:
            # 检查目标索引是否存在
            target_group = f"traj_{target_index}"
            if target_group in f:
                print(f"\n=== 内容 traj_{target_index} ===")
                f.visititems(lambda name, obj: print_hdf5_item(name, obj, target_index))
            else:
                print(f"[ERROR] 索引 traj_{target_index} 不存在!")
                # 显示可用的索引
                available_indices = []
                for key in f.keys():
                    if key.startswith('traj_'):
                        try:
                            idx = int(key.split('_')[1])
                            available_indices.append(idx)
                        except:
                            continue
                if available_indices:
                    print(f"[INFO] 可用的索引: {sorted(available_indices)}")
                else:
                    print("[INFO] 文件中没有找到 traj_* 组")
        else:
            # 打印所有内容
            f.visititems(print_hdf5_item)
    
    print("[INFO] Done.\n")

def get_available_indices(filename):
    """获取文件中所有可用的轨迹索引"""
    try:
        with h5py.File(filename, 'r') as f:
            indices = []
            for key in f.keys():
                if key.startswith('traj_'):
                    try:
                        idx = int(key.split('_')[1])
                        indices.append(idx)
                    except:
                        continue
            return sorted(indices)
    except Exception as e:
        print(f"[ERROR] 无法读取文件: {e}")
        return []

# 使用示例
if __name__ == "__main__":
    # hdf5_filename = "/mnt/pfs/3zpd5q/code/zf/raw_data/matterport_data/objnav_0_11_10655.h5"
    hdf5_filename = "/mnt/pfs/3zpd5q/code/zf/train_data/process_data/process_move/episode_2460.hdf5"
    
    # 方法1: 查看所有可用索引
    # print("=== 获取可用索引 ===")
    # indices = get_available_indices(hdf5_filename)
    # print(f"可用的轨迹索引: {indices}")
    
    # 方法2: 查看特定索引（例如 traj_0）
    # print("\n=== 查看特定索引 ===")
    # read_and_print_hdf5_file(hdf5_filename, target_index=4554)
    
    # 方法3: 查看所有内容（原始功能）
    read_and_print_hdf5_file(hdf5_filename)
    
    # 方法4: 查看多个特定索引
    # print("\n=== 查看多个索引 ===")
    # for idx in [0, 1, 2]:  # 可以修改为你想查看的索引列表
    #     read_and_print_hdf5_file(hdf5_filename, target_index=idx)
#==========================show_animation=======================================================#
# import h5py
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
# import numpy as np

# def create_animation_from_h5(hdf5_filename, output_gif_path, fps=10):
#     with h5py.File(hdf5_filename, 'r') as f:
#         episode_names = sorted(list(f.keys()))  # 所有 episode，按名字排序
#         print(f"Found {len(episode_names)} episodes")
        
#         # 收集所有帧和对应的标题
#         frames = []
#         titles = []
        
#         # 遍历所有 episode
#         for ep_idx, episode_name in enumerate(episode_names):
#             print(f"Processing {episode_name}")
            
#             episode_group = f[episode_name]
            
#             if 'image_data' not in episode_group:
#                 print(f"⚠️ {episode_name} does not contain 'image_data'")
#                 continue
            
#             # 获取图像数据
#             image_data = episode_group['image_data'][:]  # shape: [N, H, W, C]
            
#             # 获取instruction数据作为标题
#             if 'instruction' in episode_group.attrs:
#                 instruction = episode_group.attrs['instruction']
#             elif 'instruction' in episode_group:
#                 instruction = episode_group['instruction'][()]
#             else:
#                 instruction = f"Episode: {episode_name}"

#             # 确保 instruction 是字符串
#             if isinstance(instruction, bytes):
#                 instruction = instruction.decode("utf-8")

            
#             # print(f"  Number of images: {image_data.shape[0]}")
#             # print(f"  Image shape: {image_data.shape[1:]}")
#             # print(f"  Instruction: {instruction}")
            
#             # 为每个图像添加标题
#             for i, img in enumerate(image_data):
#                 frames.append(img)
#                 titles.append(f"{instruction}\n{episode_name} - Frame {i+1}/{image_data.shape[0]}")
        
#         # 创建动画
#         if not frames:
#             print("No frames to process!")
#             return
#         # print(f"Total frames collected: {len(frames)}")
#         # print(f"First frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
#         # print(f"Min: {frames[0].min()}, Max: {frames[0].max()}")
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
#         def update(frame):
#             ax.clear()
#             ax.imshow(frames[frame])
#             # print(f"Rendering frame {frame+1}/{len(frames)} - shape: {img.shape}, dtype: {img.dtype}")
#             ax.set_title(titles[frame], fontsize=10)
#             ax.axis('off')
#             return ax,
        
#         # 创建动画
#         animation = FuncAnimation(fig, update, frames=len(frames), 
#                                  interval=1000//fps, blit=False)
        
#         # 保存为GIF
#         print(f"Saving animation to {output_gif_path}...")
#         animation.save(output_gif_path, writer=PillowWriter(fps=fps), 
#                       dpi=100, progress_callback=lambda i, n: print(f"Saving frame {i+1}/{n}") if i % 10 == 0 else None)
        
#         plt.close(fig)
#         print("Animation saved successfully!")

# # 使用示例
# hdf5_filename = "/media/zhangfeng/T7/h5/objnav_data_episode_1087.h5"  # 替换为您的H5文件路径
# output_gif_path = "objnav_data_episode_1087.gif"  # 输出GIF文件路径
# create_animation_from_h5(hdf5_filename, output_gif_path, fps=10)