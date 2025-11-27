import torch
import matplotlib.pyplot as plt
import os
import numpy as np


# 创建保存文件夹
output_dir = "plot_in_train/OUTPUT=OUTPUT/qwen2_dexvln_move_and_rotate_10w_60k_4"
os.makedirs(output_dir, exist_ok=True)

def plot_actions(predicted_actions, noise_pred, target_actions, loss, step):
    # 颜色设置
    pred_color = 'tab:orange'
    target_color = 'tab:blue'
    noise_color = 'tab:red'

    # 遍历每个 batch
    for i in range(predicted_actions.shape[0]):
        pred = predicted_actions[i].to(dtype=torch.float32).cpu().numpy()
        target = target_actions[i].to(dtype=torch.float32).cpu().numpy()
        pred_noise = noise_pred[i].to(dtype=torch.float32).cpu().numpy()
        fig, (ax_xy, ax_yaw) = plt.subplots(2, 1, figsize=(8, 8))

        # ===== XY plot =====
        ax_xy.plot(target[:, 0], target[:, 1], color=target_color, label='Target XY')
        ax_xy.plot(pred[:, 0], pred[:, 1], color=pred_color, label='Predicted XY')
        # ax_xy.plot(pred_noise[:, 0], pred_noise[:, 1], color=noise_color, label='Noise XY')
        ax_xy.set_title(f'Batch {i} - XY Trajectory ----------------{loss}')
        ax_xy.set_xlabel('X')
        ax_xy.set_ylabel('Y')
        ax_xy.legend()

        # x_all = np.concatenate([pred[:, 0], target[:, 0],pred_noise[:, 0]])
        # y_all = np.concatenate([pred[:, 1], target[:, 1],pred_noise[:, 0]])
        x_all = np.concatenate([pred[:, 0], target[:, 0]])
        y_all = np.concatenate([pred[:, 1], target[:, 1]])
        # 找最大跨度
        x_range = x_all.max() - x_all.min()
        y_range = y_all.max() - y_all.min()
        max_range = max(x_range, y_range, 2.5)  # ✅ 最小不小于 0.5

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
        ax_yaw.set_title(f'Batch {i} - Yaw')
        ax_yaw.set_xlabel('Timestep')
        ax_yaw.set_ylabel('Yaw')
        ax_yaw.legend()
        ax_yaw.grid(True)

        # ===== Save figure =====
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"t_{step}_batch_{i}_comparison.png")
        plt.savefig(save_path)
        plt.close(fig)

        print(f"✅ Saved: {save_path}")
