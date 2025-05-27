import os
import h5py
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
def savitzky_golay_smooth(positions, window_size=10, poly_order=2, threshold=25):
    """
    使用Savitzky-Golay滤波器平滑轨迹，避免平滑相同的点，并保留连续30个不变点段。

    Args:
        positions (np.ndarray): 位置数据，形状 (T, 3)。
        window_size (int): 滑动窗口的大小。
        poly_order (int): 多项式阶数。
        threshold (int): 连续不变点段的长度阈值，超过这个长度的点段不进行平滑。

    Returns:
        np.ndarray: 平滑后的轨迹。
    """
    smoothed_positions = np.copy(positions)
    # 获取数据的长度
    n = len(positions)

    # 遍历每一维度
    for dim in range(3):
        column = positions[:, dim]
        
        # 检查不变的点
        unchanged = np.ones(n, dtype=bool)  # 标记所有点最初为不变
        for i in range(1, n):
            if column[i] == column[i - 1]:  # 如果相邻点相同，标记为不变
                unchanged[i] = unchanged[i - 1]  # 保持不变状态为 True

        # 寻找连续不变点的段落
        start = 0
        while start < n:
            if unchanged[start]:
                # 找到连续不变的段落
                end = start
                while end < n and unchanged[end]:
                    end += 1

                # 如果该段不变的长度大于阈值，保留这一段
                if end - start >= threshold:
                    smoothed_positions[start:end, dim] = column[start:end]  # 不变段直接保留

                # 处理平滑段
                else:
                    smoothed_positions[start:end, dim] = savgol_filter(column[start:end], window_size, poly_order)
                
                # 继续从 end 开始检查
                start = end
            else:
                # 如果当前位置是变动的，直接平滑
                smoothed_positions[start, dim] = savgol_filter(column[start:start+1], window_size, poly_order)
                start += 1
    return smoothed_positions


def group_indices_by_string(strings):
    """
    Args:
        strings (List[str]): 一组字符串，例如 new_annotations
    Returns:
        Dict[str, List[int]]: 键是字符串内容，值是它出现的所有索引
    """
    group_map = defaultdict(list)
    for i, s in enumerate(strings):
        group_map[s].append(i)
    return group_map
def interpolate_annotation_indices(num_annotations, num_interp):
    assert num_interp == (num_annotations - 1) * 10 + 1, "插值长度应为 (N-1)*10 + 1"

    index_mapping = []
    for i in range(num_annotations - 1):
        index_mapping += [i] * 10  # 每段插值10帧
    index_mapping.append(num_annotations - 1)  # 最后一帧

    return np.array(index_mapping)
def process_single_hdf5_file(h5_file_path):
    if not os.path.exists(h5_file_path):
        print(f"文件不存在: {h5_file_path}")
        return

    filename = os.path.basename(h5_file_path)
    folder_name = filename.split('.')[0]  # 使用文件名（去掉扩展名）作为文件夹名
    folder_path = os.path.join(os.path.dirname(h5_file_path), folder_name)
    os.makedirs(folder_path, exist_ok=True)

    try:
        with_timestep = False
        with h5py.File(h5_file_path, 'r') as f:
            instruction = f['instruction'][()].decode('utf-8')
            new_annotations = f['annotations_long2'][()]
            # new_annotations = f['annotations_action'][()]
            type_data = f['type'][()].decode('utf-8')
            rel_pos = f['relative_pos'][:]  # 位置数据，形状 (T, 3)
            rel_yaw = f['relative_yaw'][:]  # 旋转数据，形状 (T, 3)
            rel_pos = -np.round(rel_pos, 2)
            for i in range(len(new_annotations)):

                k = i*10
                pos = rel_pos[k]
                yaw = rel_yaw[k]
                #smooth
                pos = savitzky_golay_smooth(pos)


                xs = pos[:, 0]
                ys = pos[:, 1]
                zs = pos[:, 2]

                # 创建图形
                fig, axes = plt.subplots(1, 2, figsize=(12, 12))

                if with_timestep:
                # 3D 图，展示第一帧相对位置
                
                    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                    
                    ts = np.arange(len(xs)) 

                    ax1.scatter(xs, zs, ts, s=10, c=ts, cmap='viridis', label='Trajectory (Time-encoded)')

                    ax1.invert_yaxis() 
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    ax1.set_zlabel('timestep')
                    ax1.set_title(new_annotations[i])
                else:
                    ax1 = axes[0]

                    ax1.scatter(xs, zs, label='Trajectory (Time-encoded)')

                    ax1.invert_yaxis() 
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    ax1.set_title(new_annotations[i])

                # 设置相同的尺度：通过手动设置轴的范围
                x_range = [min(xs), max(xs)]
                # y_range = [min(ys), max(ys)]
                z_range = [min(zs), max(zs)]

                # 计算最大范围
                # max_range = max(max(x_range[1] - x_range[0], y_range[1] - y_range[0]), z_range[1] - z_range[0])
                max_range = max(x_range[1] - x_range[0], z_range[1] - z_range[0])

                # 设置每个轴的范围
                ax1.set_xlim([min(x_range) - max_range * 0.1, max(x_range) + max_range * 0.1])
                ax1.set_ylim([min(z_range) - max_range * 0.1, max(z_range) + max_range * 0.1])
                # ax1.set_zlim([min(z_range) - max_range * 0.1, max(z_range) + max_range * 0.1])


                # 2D 图，展示第一帧的偏航角度
                ax2 = axes[1]
                ax2.plot(range(len(yaw)), yaw, label='yaws', color='g')

                ax2.set_xlabel('Frame Index')
                ax2.set_ylabel('yaw')
                ax2.set_title('Yaws')


                output_path = f"z_images/rel_{i}.png"  # 自定义路径
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)  # 释放资源，防止内存泄露

                # break
            print("down")

    except Exception as e:
        print(f"处理文件 {h5_file_path} 时出错: {e}")


# 示例调用
h5_path = "data/sample/episode_1.h5"
process_single_hdf5_file(h5_path)
