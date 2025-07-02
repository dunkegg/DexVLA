import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

# 创建保存文件夹
output_dir = "test_data/visulization3"
os.makedirs(output_dir, exist_ok=True)

def plot_actions(i,predicted_actions, target_actions, raw_lang, post_process, frames):
    save_path = os.path.join(output_dir, f"test_{i}")

    os.makedirs(save_path, exist_ok=True)  

    for idx, img in enumerate(frames):
        out_path = os.path.join(save_path, f"frame_{idx:04d}.png")  # 命名格式: frame_0000.png
        cv2.imwrite(out_path, img)

    # out_path = os.path.join(output_dir, f"case_{i}.png")  # 命名格式: frame_0000.png
    # cv2.imwrite(out_path, frames[0])

    # 颜色设置
    pred_color = 'tab:orange'
    target_color = 'tab:blue'

    new_target_actions = [post_process(raw_action) for raw_action in target_actions]
    target_actions = new_target_actions
    # 遍历每个 batch
    # for i in range(predicted_actions.shape[0]):
    pred = predicted_actions.to(dtype=torch.float32).cpu().numpy()
    target = np.array(target_actions, dtype=np.float32)
    fig, (ax_xy, ax_yaw) = plt.subplots(2, 1, figsize=(8, 8))

    # Trans =================
    target_base = target[0, :2]  # 取出第一个点的 x, y，shape: (2,)
    pred_base = pred[0, :2]

    # 拷贝一份防止覆盖原 yaw
    target_xy = target[:, :2] - target_base
    target_yaw = target[:, 2:]

    pred_xy = pred[:, :2] - pred_base
    pred_yaw = pred[:, 2:]

    # 组合回原来的形状
    target = np.concatenate([target_xy, target_yaw], axis=1)
    pred = np.concatenate([pred_xy, pred_yaw], axis=1)


    # target[:, 0] = -target[:, 0]
    # pred[:, 0] = -pred[:, 0]
    # pred[:, 1] = -pred[:, 1]

    # ===== XY plot =====
    ax_xy.plot(target[:, 0], target[:, 1], color=target_color, label='Target XY')
    ax_xy.plot(pred[:, 0], pred[:, 1], color=pred_color, label='Predicted XY')

    ax_xy.set_title(f'{raw_lang}')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.legend()


    x_all = np.concatenate([pred[:, 0], target[:, 0]])
    y_all = np.concatenate([pred[:, 1], target[:, 1]])
    # 找最大跨度
    x_range = x_all.max() - x_all.min()
    y_range = y_all.max() - y_all.min()
    max_range = max(x_range, y_range, 0.5)  # ✅ 最小不小于 0.5

    # 以中心为基础构建正方形范围
    x_center = (x_all.max() + x_all.min()) / 2
    y_center = (y_all.max() + y_all.min()) / 2

    x_min = x_center - max_range / 2
    x_max = x_center + max_range / 2
    y_min = y_center - max_range / 2
    y_max = y_center + max_range / 2

    # 应用到坐标轴
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True)



    # ===== Yaw plot =====
    ax_yaw.plot(target[:, 2], color=target_color, label='Target Yaw')
    ax_yaw.plot(pred[:, 2], color=pred_color, label='Predicted Yaw')
    ax_yaw.set_title(f'Yaw')
    ax_yaw.set_xlabel('Timestep')
    ax_yaw.set_ylabel('Yaw')
    ax_yaw.legend()
    ax_yaw.grid(True)

    # ===== Save figure =====
    plt.tight_layout()
    save_path = os.path.join(save_path, f"actions.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"✅ Saved: {save_path}")
