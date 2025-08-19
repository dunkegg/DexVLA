import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import os
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
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


def plot_obs(time,predicted_actions ,raw_lang, obs,human_position, target_actions = None, smooth_actions = None):
    # for idx, img in enumerate(frames):
    #     out_path = os.path.join(output_dir, f"case_{i}_frame_{idx:04d}.png")  # 命名格式: frame_0000.png
    #     cv2.imwrite(out_path, img)


    # 颜色设置
    pred_color = 'tab:blue'
    target_color = 'tab:red'
    smooth_color = 'tab:green'

    # 遍历每个 batch
    # for i in range(predicted_actions.shape[0]):
    # pred = predicted_actions.to(dtype=torch.float32).cpu().numpy()
    # pred = [post_process(raw_action) for raw_action in pred]
    pred = predicted_actions
    pred_base = pred[0, :2]


    pred_xy = pred[:, :2] - pred_base
    if human_position is not None:
        human_position[0] = human_position[0] - pred_base[0]
        human_position[2] = human_position[2] - pred_base[1]
    pred_yaw = pred[:, 2:]

    # 组合回原来的形状

    pred = np.concatenate([pred_xy, pred_yaw], axis=1)

    # === 创建大图 ===
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig) 
    # 左侧：obs 图像
    ax_img = fig.add_subplot(1, 2, 1)
    # ax_img.imshow(cv2.cvtColor(obs['color_0_0'], cv2.COLOR_BGR2RGB))
    if isinstance(obs, Image.Image):
        obs = np.array(obs)  # PIL -> RGB numpy
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # 如果后续逻辑假设 BGR
    ax_img.imshow(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
    ax_img.set_title('Observation')
    ax_img.axis('off')

    # 右侧：动作轨迹图
    ax_xy = fig.add_subplot(1, 2, 2)
    # ax_yaw = fig.add_subplot(2, 2, 4)

    ax_xy.plot(pred[:, 0], pred[:, 1], color=pred_color, label='Predicted XY')
    ax_xy.plot(pred[:16, 0], pred[:16, 1], color='purple', label='Early Steps (0~15)', linewidth=2)
    ax_xy.plot(pred[:12, 0], pred[:12, 1], color='yellow', label='Early Steps (0~11)', linewidth=2)
    if target_actions is not None:
        ax_xy.plot(target_actions[:, 0], target_actions[:, 1], color=target_color, label='Target XY')
    if smooth_actions is not None:
        base = smooth_actions[0, :2]
        smooth_xy = smooth_actions[:, :2] - base
        ax_xy.plot(smooth_xy[:, 0], smooth_xy[:, 1], color=smooth_color, label='Smooth XY')
    if human_position is not None:
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
    max_range = 3
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
    
    # out_path = os.path.join(output_dir, f"{time}.png")  # 命名格式: frame_0000.png
    # # cv2.imwrite(out_path, obs)
    # plt.savefig(out_path)
    # plt.close(fig)

    # print(f"✅ Saved: {out_path}")
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_np = np.asarray(buf)[:, :, :3]  # 去掉 alpha 通道（RGB）
    plt.close(fig)

    return img_np  # 返回 numpy 格式图像

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from io import BytesIO
from PIL import Image
import math

def plot_ctrl(now_time, world_actions, pid_pos, agent_pos, followed_pos, origin_pos,cur_image):
    """
    Args:
        now_time: 当前时间
        world_actions: (30,4) array, 每个是 [x, height, y, yaw]
        pid_pos: [x, y, yaw] 当前 PID 输出目标
        agent_pos: [x, y, yaw] 当前机器人实际位置
        followed_pos: [x, y] 被跟随的位置
        cur_image: numpy图像 (H,W,3)
    Returns:
        numpy array (合并后的图)
    """
    yaw_bias = math.pi / 2

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # --- 左边放原图 ---
    axs[0].imshow(cur_image)
    axs[0].axis('off')
    axs[0].set_title(f"Camera View t={now_time:.1f}s")

    # --- 右边画轨迹 ---
    # 以 agent_pos 为原点
    traj_x = world_actions[:, 0] - origin_pos[0]
    traj_y = world_actions[:, 2] - origin_pos[1]
    axs[1].plot(traj_x, -traj_y, 'b-', label="Planned Path")
    # axs[1].scatter(traj_x[0], -traj_y[0], c='blue', marker='o', label="Start")

    # 当前机器人位置
    axs[1].scatter(0, 0, c='black', s=50, label="Origin")
    # 朝向箭头
    arrow_len = 0.3

    agent_dx = agent_pos[0] - origin_pos[0]
    agent_dy = agent_pos[1] - origin_pos[1]
    axs[1].scatter(agent_dx, -agent_dy, c='green', s=30, label="Agent")
    axs[1].arrow(agent_dx, -agent_dy,
                 arrow_len * np.cos(agent_pos[2]+yaw_bias),
                 arrow_len * np.sin(agent_pos[2]+yaw_bias),
                 head_width=0.1, head_length=0.1, fc='green', ec='green')

    # PID 目标位置 + 朝向
    pid_dx = pid_pos[0] - origin_pos[0]
    pid_dy = pid_pos[1] - origin_pos[1]
    axs[1].scatter(pid_dx, -pid_dy, c='orange', s=30, label="PID Target")
    axs[1].arrow(pid_dx, -pid_dy,
                 arrow_len * np.cos(pid_pos[2]+yaw_bias),
                 arrow_len * np.sin(pid_pos[2]+yaw_bias),
                 head_width=0.1, head_length=0.1, fc='orange', ec='orange')

    # 跟随点
    follow_dx = followed_pos[0] - origin_pos[0]
    follow_dy = followed_pos[1] - origin_pos[1]
    axs[1].scatter(follow_dx, -follow_dy, c='red', s=50, label="Followed Pos")

    axs[1].axis('equal')
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Y (m)")
    axs[1].legend()
    axs[1].set_title("Control Visualization (Top-down)")

    # 把图转成 numpy
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)
    return img_np