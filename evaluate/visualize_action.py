import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import os
import numpy as np
import cv2

# 创建保存文件夹
output_dir = "evaluate/plot_action"
os.makedirs(output_dir, exist_ok=True)

def plot_actions(i,predicted_actions, target_actions, raw_lang, post_process, frames):
    # for idx, img in enumerate(frames):
    #     out_path = os.path.join(output_dir, f"case_{i}_frame_{idx:04d}.png")  # 命名格式: frame_0000.png
    #     cv2.imwrite(out_path, img)

    out_path = os.path.join(output_dir, f"case_{i}.png")  # 命名格式: frame_0000.png
    cv2.imwrite(out_path, frames[0])

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


    target[:, 0] = -target[:, 0]
    pred[:, 0] = -pred[:, 0]
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
    save_path = os.path.join(output_dir, f"case_{i}_actions.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"✅ Saved: {save_path}")


def plot_obs(time,predicted_actions, post_process, raw_lang, obs,human_position):
    # for idx, img in enumerate(frames):
    #     out_path = os.path.join(output_dir, f"case_{i}_frame_{idx:04d}.png")  # 命名格式: frame_0000.png
    #     cv2.imwrite(out_path, img)


    # 颜色设置
    pred_color = 'tab:blue'
    # target_color = 'tab:blue'

    # 遍历每个 batch
    # for i in range(predicted_actions.shape[0]):
    # pred = predicted_actions.to(dtype=torch.float32).cpu().numpy()
    # pred = [post_process(raw_action) for raw_action in pred]
    pred = predicted_actions
    pred_base = pred[0, :2]


    pred_xy = pred[:, :2] - pred_base
    pred_yaw = pred[:, 2:]

    # 组合回原来的形状

    pred = np.concatenate([pred_xy, pred_yaw], axis=1)

    # === 创建大图 ===
    fig = plt.figure(figsize=(12, 6))
    
    # 左侧：obs 图像
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(cv2.cvtColor(obs['color_0_0'], cv2.COLOR_BGR2RGB))
    ax_img.set_title('Observation')
    ax_img.axis('off')

    # 右侧：动作轨迹图
    ax_xy = fig.add_subplot(1, 2, 2)
    # ax_yaw = fig.add_subplot(2, 2, 4)

    ax_xy.plot(pred[:, 0], pred[:, 1], color=pred_color, label='Predicted XY')
    ax_xy.scatter(human_position[0], human_position[2], color='red', s=50, label='End Point')
    ax_xy.set_title(raw_lang)
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.legend()
    ax_xy.grid(True)
    ax_xy.set_aspect('equal')

    # 设置 XY 统一视角范围
    x_all = pred[:, 0]
    y_all = pred[:, 1]
    max_range = 8
    x_center = (x_all.max() + x_all.min()) / 2
    y_center = (y_all.max() + y_all.min()) / 2
    ax_xy.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax_xy.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    # # ===== Yaw plot =====
    # ax_yaw.plot(pred[:, 2], color=pred_color, label='Predicted Yaw')
    # ax_yaw.set_title(f'Yaw')
    # ax_yaw.set_xlabel('Timestep')
    # ax_yaw.set_ylabel('Yaw')
    # ax_yaw.legend()
    # ax_yaw.grid(True)

    # ===== Save figure =====
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"{time}.png")  # 命名格式: frame_0000.png
    # cv2.imwrite(out_path, obs)
    plt.savefig(out_path)
    plt.close(fig)

    print(f"✅ Saved: {out_path}")
