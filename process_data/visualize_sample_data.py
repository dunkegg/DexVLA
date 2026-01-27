
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import cv2
import PIL.Image as PILImage

def visualize_trajectory(
    cv_image,
    pixel_coords,
    instruction="",
    supervised_action=None,
    subreason=None
):
    if cv_image is None:
        return None

    # resize
    target_w, target_h = 960, 720
    # img = cv2.resize(cv_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = cv_image
    h, w, _ = img.shape
    center_x, center_y = w // 2, h - 1

    # ---------- 字幕 ----------
    subtitle_h = h // 7
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, subtitle_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, subtitle_h / 350)
    thickness = max(1, int(font_scale * 2))

    if instruction:
        cv2.putText(
            img, instruction, (15, int(subtitle_h * 0.6)),
            font, font_scale, (255, 255, 255),
            thickness, cv2.LINE_AA
        )

    if subreason:
        cv2.putText(
            img, subreason, (15, 20),
            font, font_scale, (255, 255, 255),
            thickness, cv2.LINE_AA
        )

    # ---------- 轨迹 ----------
    scale = 70.0
    overlay = img.copy()

    if supervised_action is not None:
        for i in range(len(supervised_action) - 1):
            x1, y1 = supervised_action[i, :2]
            x2, y2 = supervised_action[i + 1, :2]
            y1 = -y1
            y2 = -y2
            p1 = (int(center_x + y1 * scale),
                  int(center_y - x1 * scale))
            p2 = (int(center_x + y2 * scale),
                  int(center_y - x2 * scale))

            cv2.line(overlay, p1, p2, (0, 255, 100), 3, cv2.LINE_AA)

    # ---------- 像素点 ----------
    if pixel_coords is not None:
        # 统一成 list[(u, v), ...]
        if isinstance(pixel_coords, (list, tuple, np.ndarray)) and len(pixel_coords) > 0 \
        and isinstance(pixel_coords[0], (list, tuple, np.ndarray)):
            coords = pixel_coords
        else:
            coords = [pixel_coords]

        for (u, v) in coords:
            cv2.circle(
                overlay,
                (int(u), int(v)),
                8,
                (255, 0, 0),  # 蓝色（BGR）
                -1
            )

    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    return img



def plot_topdown_and_yaw(
    actions: np.ndarray,        # (T,3) [x, y, yaw]
    human_local: np.ndarray,    # (3,)
    ax_top,
    ax_bot,
    cmap_name="viridis"
):
    # =========================
    # 数据准备
    # =========================
    traj_xy = actions[:, :2]               # (T,2) x-forward, y-left
    yaw_deg = np.degrees(actions[:, 2])    # Δyaw (deg)
    steps   = np.arange(len(actions))

    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=0, vmax=len(actions)-1)
    colors_arr = cmap(norm(steps))

    # =========================
    # 上：平面渐变轨迹
    # =========================
    segs = np.concatenate(
        [traj_xy[:-1, None, :], traj_xy[1:, None, :]],
        axis=1
    )
    lc = LineCollection(segs, colors=colors_arr[:-1], linewidths=2)
    ax_top.add_collection(lc)

    # 起点 / 人
    ax_top.scatter(0, 0, c="red", marker="*", s=120, label="ego (0,0)")
    ax_top.scatter(
        human_local[0], human_local[1],
        c="blue", marker="*", s=120, label="human_local"
    )

    ax_top.set_aspect("equal")
    ax_top.set_xlim(-5, 5)
    ax_top.set_ylim(-5, 5)
    # ax_top.set_xlabel("x (forward, m)")
    # ax_top.set_ylabel("y (left, m)")
    ax_top.set_xlabel("x (forward, m)")
    ax_top.set_ylabel("y (left, m)")
    ax_top.legend(fontsize="small")
    ax_top.set_title("Top-down trajectory")

    # =========================
    # 下：yaw 渐变曲线
    # =========================
    for i in range(len(yaw_deg) - 1):
        ax_bot.plot(
            steps[i:i+2],
            yaw_deg[i:i+2],
            color=colors_arr[i],
            linewidth=2
        )

    ax_bot.set_xlabel("timestep")
    ax_bot.set_ylabel("Δyaw (deg)")
    ax_bot.grid(True, alpha=0.3)

def plot_sample_data(cv_image, pixel_coords, instruction, actions, human_local, save_path,subreason=None):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    # ========== 左：图像 ==========
    ax_img = fig.add_subplot(gs[0])
    img = visualize_trajectory(
        cv_image,
        pixel_coords,
        instruction,
        actions,
        subreason
    )
    # ax_img.imshow(img[:, :, ::-1])  # BGR → RGB
    ax_img.imshow(img)  # BGR → RGB
    ax_img.axis("off")
    if pixel_coords is None:
        ax_img.set_title("RGB + trajectory/ No pixel goal")
    else:
        ax_img.set_title("RGB + trajectory")

    # ========== 右：平面 ==========
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[2, 1])
    ax_top = fig.add_subplot(gs_right[0])
    ax_bot = fig.add_subplot(gs_right[1])

    plot_topdown_and_yaw(
        actions=actions,
        human_local=human_local,
        ax_top=ax_top,
        ax_bot=ax_bot
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def render_sequence_and_make_video(output,frame_dir,video_path,fps=2):
    cv_images = output['obs']
    follow_data = output['follow_paths']
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for i in range(len(cv_images)):
        frame_path = frame_dir / f"frame_{i:05d}.png"
        depth_path = frame_dir / f"frame_{i:05d}_depth.png"
        cur_data = follow_data[i]
        rgb, depth = cv_images[i]
        # depth_vis = np.clip(depth, 0, 10)
        depth_vis = (depth / 10.0 * 255).astype(np.uint8)
        # path = os.path.join("rebuild_habitat", "depth_vis", out_name)
        PILImage.fromarray(depth_vis).save(depth_path)
        plot_sample_data(
            cv_image=rgb,
            pixel_coords=cur_data['pixel_coords'],
            instruction=cur_data['desc'],
            actions=cur_data['actions'],
            human_local=cur_data['relative_human_state'],
            subreason=None,
            save_path=frame_path
        )

        frame_paths.append(frame_path)

    # video_path = Path(video_path)
    # video_path.parent.mkdir(parents=True, exist_ok=True)

    # first = cv2.imread(str(frame_paths[0]))
    # h, w, _ = first.shape

    # writer = cv2.VideoWriter(
    #     str(video_path),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (w, h)
    # )

    # for p in frame_paths:
    #     img = cv2.imread(str(p))
    #     writer.write(img)

    # writer.release()
def visualize_follow_path(group: h5py.Group,
                          actions: np.ndarray,
                          human_local,
                          out_png: Path,
                          cmap_name: str = "viridis"):
    """
    actions : (T,3)  [x, z, yaw]   —— 已经是局部坐标
    human_local : (3,)
    """
    # ------ 数据准备 -----------------------------------------------------
    traj_xz = actions[:, :2]                       # (T,2)  x,z
    yaw_deg = np.degrees(actions[:, 2])            # (T,)   yaw°
    steps   = np.arange(len(actions))              # 0..T-1

    # 生成归一化颜色映射器
    cmap   = cm.get_cmap(cmap_name)
    norm   = colors.Normalize(vmin=0, vmax=len(actions)-1)
    colors_arr = cmap(norm(steps))

    # ------ 画布 ---------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=False
    )

    # ── 上：渐变轨迹 ───────────────────────────────────────────────
    # 把折线拆成线段集合，用 LineCollection 着色
    segs = np.concatenate(
        [traj_xz[:-1, None, :], traj_xz[1:, None, :]], axis=1
    )
    lc   = LineCollection(segs, colors=colors_arr[:-1], linewidths=2)
    ax_top.add_collection(lc)
    ax_top.scatter(0, 0, c="red", marker="*", s=100, label="follow (0,0)")
    ax_top.scatter(human_local[0], human_local[2],
                   c="blue", marker="*", s=100, label="human_local")
    ax_top.set_aspect("equal")
    ax_top.set_xlim(-5, 5); ax_top.set_ylim(-5, 5)
    ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("z (m)")
    ax_top.legend(fontsize="small")
    ax_top.set_title(f"obs_idx = {int(group['obs_idx'][()])}")

    # ── 下：Δyaw 渐变曲线 ─────────────────────────────────────────
    for i in range(len(yaw_deg)-1):
        ax_bot.plot(steps[i:i+2], yaw_deg[i:i+2],
                    color=colors_arr[i], linewidth=2)
    ax_bot.set_xlabel("timestep")
    ax_bot.set_ylabel("Δyaw (deg)")
    ax_bot.grid(True, alpha=0.3)

    # colorbar (可选)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.set_label("time step")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)