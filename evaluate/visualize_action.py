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
    # ax_xy.plot(pred[:12, 0], pred[:12, 1], color='yellow', label='Early Steps (0~11)', linewidth=2)
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


def visualize_trajectory(cv_image, all_actions, instruction="", surpervised_action=None, subreason = None):
    if cv_image is None or all_actions is None:
        return

    # ============================================================
    # 1) 先把图片 resize 到 (960, 720)
    # ============================================================
    target_w, target_h = 960, 720
    img = cv2.resize(cv_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    h, w, _ = img.shape
    center_x, center_y = w // 2, h - 1

    # ============================================================
    # 2) 字幕绘制 —— 右上角，宽度一半，高度 1/3
    # ============================================================
    subtitle_w = w // 3        # 黑背景宽度：右半部分
    subtitle_h = h // 7        # 高度占整图高度的 1/3

    overlay = img.copy()

    # 黑底区域（右上角）
    cv2.rectangle(
        overlay,
        (0, 0),     # 右上角
        (w, subtitle_h),         # 右上角向下
        (0, 0, 0),
        -1
    )

    # 半透明融合
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    # ============================================================
    # 3) 字幕文本 —— 右对齐，两行
    # ============================================================
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, subtitle_h / 350)
    thickness = max(1, int(font_scale * 2))

    # 将指令分两行
    half = len(instruction) // 2
    line1 = instruction[:half]
    line2 = instruction[half:]

    def draw_right_text(img, text, y):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = w - text_size[0] - 15  # 右对齐（右边留 15px）
        cv2.putText(img, text, (15, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


    draw_right_text(img, instruction, int(subtitle_h * 0.5))
    # draw_right_text(img, line2, int(subtitle_h * 0.66))

    if subreason is not None:
        cv2.putText(img, subreason, (15, 15), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA)

    # ============================================================
    # 4) 绘制轨迹
    # ============================================================
    scale = 70.0

    overlay = img.copy()
    for i in range(len(all_actions) - 1):
        x1, y1 = all_actions[i, :2]
        x2, y2 = all_actions[i + 1, :2]
        p1 = (int(center_x + x1 * scale), int(center_y - y1 * scale))
        p2 = (int(center_x + x2 * scale), int(center_y - y2 * scale))
        cv2.line(overlay, p1, p2, (255, 80, 0), 3, lineType=cv2.LINE_AA)

    if surpervised_action is not None:
        for i in range(len(surpervised_action) - 1):
            x1, y1 = surpervised_action[i, :2]
            x2, y2 = surpervised_action[i + 1, :2]
            p1 = (int(center_x + x1 * scale), int(center_y - y1 * scale))
            p2 = (int(center_x + x2 * scale), int(center_y - y2 * scale))
            cv2.line(overlay, p1, p2, (0, 255, 100), 3, lineType=cv2.LINE_AA)

    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    return img




def plot_ctrl(now_time,future_local, world_actions, cmd_pos, human_pos, cur_image, stop):
    """
    Args:
        now_time: float，当前时间
        world_actions: list of (pos, quat, yaw)
                       pos = [x,y,z]
        cmd_pos: [x, y, z]
        human_pos: [x, y, z]
        cur_image: numpy图像 (H,W,3)
    """
    # future_local = np.delete(future_local, 1, axis=1)
    img_with_local = visualize_trajectory(np.array(cur_image) , future_local)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # -------------------- 左边：原图 --------------------
    axs[0].imshow(img_with_local)
    axs[0].axis('off')
    axs[0].set_title(f"Camera View t={now_time:.1f}s")

    # -------------------- 右边：轨迹图 --------------------

    ax = axs[1]
    if not stop:
        # world_actions 里的轨迹（只取 pos.x / pos.z）
        traj = np.array([np.array([p[0].x, p[0].z]) for p in world_actions])
        ax.plot(traj[:, 0], traj[:, 1], '-o', markersize=3, label="Trajectory (pos.xz)")

        # 当前 cmd_pos
        ax.scatter(cmd_pos[0], cmd_pos[2], s=80, c='red', label="cmd_pos")

    # 当前 human_pos
    ax.scatter(human_pos[0], human_pos[2], s=80, c='blue', label="human_pos")

    ax.set_title("World X–Z Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend()
    ax.grid(True)

    # 把图转成 numpy
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)
    return img_np