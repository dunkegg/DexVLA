import os
import h5py
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
def savitzky_golay_smooth(positions, window_size=100, poly_order=2, threshold=30):
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
        with h5py.File(h5_file_path, 'r') as f:
            instruction = f['instruction'][()].decode('utf-8')
            # new_annotations = f['annotations_long2'][()]
            new_annotations = f['annotations_action'][()]
            type_data = f['type'][()].decode('utf-8')
            abs_pos = f['abs_pos'][:]  # 位置数据，形状 (T, 3)
            abs_rot = f['abs_rot'][:]  # 旋转数据，形状 (T, 3)

            #smooth
            # abs_pos = savitzky_golay_smooth(abs_pos)


            # 计算以第一帧为原点的相对位置
            first_pos = abs_pos[0]  # 第一帧的绝对位置
            relative_abs_pos = abs_pos - first_pos  # 所有帧相对于第一帧的位置

            horizon = 50

            xs = relative_abs_pos[:, 0]
            ys = relative_abs_pos[:, 1]
            zs = relative_abs_pos[:, 2]

            # 插值映射
            index_mapping = interpolate_annotation_indices(len(new_annotations), len(relative_abs_pos[:, 0]))

            # 分组
            groups = group_indices_by_string(new_annotations)

            # 为每个组分配颜色
            color_map = get_cmap("tab10")  # 或者 'tab20'
            string_to_color = {string: color_map(i) for i, string in enumerate(groups)}

            # 画图
            fig = plt.figure()
            # ax1 = fig.add_subplot(111, projection='3d')
            ax1 = fig.add_subplot(111)

            for i in range(len(xs)):
                orig_idx = index_mapping[i]
                label = new_annotations[orig_idx]
                color = string_to_color[label]
                # ax1.scatter(xs[i], ys[i], zs[i], color=color)
                ax1.scatter(xs[i], zs[i], color=color)

            legend_patches = []
            for label, color in string_to_color.items():
                patch = Patch(color=color, label=label.decode('utf-8'))
                legend_patches.append(patch)

            # 将图例放在右边
            ax1.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
            ax1.invert_yaxis() 

            output_path = "z_trajectory_action.png"  # 自定义路径
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 释放资源，防止内存泄露
            print("down")

    except Exception as e:
        print(f"处理文件 {h5_file_path} 时出错: {e}")


# 示例调用
h5_path = "data/sample/episode_1.h5"
process_single_hdf5_file(h5_path)
