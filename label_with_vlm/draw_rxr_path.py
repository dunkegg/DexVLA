# import h5py
import matplotlib.pyplot as plt
import numpy as np
from action_utils import extract_data, quat_to_yaw
from action_label import get_actions_from_direction_precise

hdf5_file = "data/raw_data/rxr_smooth/episode_52.hdf5"


segments = extract_data(hdf5_file)
# print("segment:\n",segments[:10])

""" 整个 episode 轨迹 """
# follower
follower_all = segments[0]["follow_pos"]
# print("follower_all:\n", follower_all)
follower_quat = segments[0]["follow_quat"]
# print("follower_quat:\n", follower_quat)

# 只取第0维和第2维
follower_xz = follower_all[:, [0, 2]]
follower_xz[:, 1] = -follower_xz[:, 1]  # z轴数据取相反值
# print("follower_xz:\n",follower_xz)
# print(len(follower_xz))
yaw_est_follower = quat_to_yaw(follower_quat)
# yaw_est_follower = estimate_yaw_by_neighbor_points(follower_xz)
# print("yaw_est_follower:\n",yaw_est_follower)
# print(len(yaw_est_follower))


def draw_yaw_actions(follower_xz, yaw_est_follower, actions):
    """
    在轨迹上绘制 yaw 朝向，并在箭头位置标注动作文字。
    """
    plt.figure(figsize=(10, 8))
    # Follower
    # follower 起点
    plt.scatter(follower_xz[0, 0], follower_xz[0, 1], c="blue", s=80, marker="o")
    plt.text(
        follower_xz[0, 0],
        follower_xz[0, 1],
        "  Follower Start",
        color="blue",
        fontsize=12,
        verticalalignment="bottom",
    )
    # 绘制轨迹
    plt.plot(follower_xz[:, 0], follower_xz[:, 1], label="Follower", linewidth=2)
    # 绘制yaw箭头
    arrow_step = max(len(follower_xz) // 35, 1)  # 间隔
    # print("arrow_step:",arrow_step)
    for i in range(0, len(follower_xz), arrow_step):
        x, z = follower_xz[i]
        yaw = -yaw_est_follower[i]
        dx = 0.3 * np.sin(yaw)  # 0.3：长度，sin：方向
        dz = 0.3 * np.cos(yaw)  # 0.3：长度，cos：方向
        # ---- 绘制箭头 ----
        plt.arrow(
            x,
            z,
            dx,
            dz,
            head_width=0.15,
            head_length=0.15,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )
        # ---- 绘制动作文字 ----
        if (
            i < len(actions)
            and actions[i] != "go_forward"
            and actions[i] != "approaching_final_point"
        ):
            plt.text(
                x + 0.1,
                z + 0.1,  # 文字偏移避免遮挡箭头
                actions[i],
                fontsize=8,
                color="red",
                alpha=0.8,
            )
    # plt.legend()
    # plt.axis("equal")
    # plt.grid(True)
    # plt.show()


# actions1 = get_actions_from_direction_precise(
#     follower_xz,
#     yaw_est_follower,
#     far_dist=1.5,
#     near_dist=0.5,
#     slight_deg=20.0,  # slight_deg ≈ 25° ~ 35°
#     sharp_deg=45.0,  # sharp_deg ≈ 55° ~ 70°
# )
# draw_yaw_actions(follower_xz, yaw_est_follower, actions=actions1)


def test_param_sets(follower_xz, yaw_est_follower, param_list):
    """
    批量测试多组参数，逐组绘制动作标注图。
    param_list: 列表，每个元素为 dict,如:
        {
            "far_dist": 1.5,
            "near_dist": 0.5,
            "slight_deg": 20,
            "sharp_deg": 45
        }
    """
    for idx, params in enumerate(param_list):
        print(f"\n========== 测试组合 {idx + 1} ==========")
        print(params)

        actions = get_actions_from_direction_precise(
            follower_xz,
            yaw_est_follower,
            far_dist=params["far_dist"],
            near_dist=params["near_dist"],
            slight_deg=params["slight_deg"],  # 25° ~ 35°
            sharp_deg=params["sharp_deg"],  # 55° ~ 70°
        )
        # 统计 turn_right_slightly 动作的数量
        count_turn_right_slightly = actions.count("turn_right_slightly")

        # 统计 turn_right 动作的数量
        count_turn_right = actions.count("turn_right")

        # 打印结果
        print(f"turn_right_slightly 动作的数量: {count_turn_right_slightly}")
        print(f"turn_right 动作的数量: {count_turn_right}")

        # 如果需要总数
        total_count = count_turn_right_slightly + count_turn_right
        print(f"两种动作的总数量: {total_count}")

        draw_yaw_actions(follower_xz, yaw_est_follower, actions=actions)

        # slight = np.deg2rad(params["slight_deg"])
        # print("slight:", slight)
        # sharp = np.deg2rad(params["sharp_deg"])
        # print("sharp:", sharp)

        plt.title(f"Param Set {idx + 1}: {params}")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()


param_list = [
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 25, "sharp_deg": 55},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 30, "sharp_deg": 55},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 35, "sharp_deg": 55},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 25, "sharp_deg": 60},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 30, "sharp_deg": 60},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 35, "sharp_deg": 60},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 25, "sharp_deg": 65},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 30, "sharp_deg": 65},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 35, "sharp_deg": 65},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 25, "sharp_deg": 70},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 30, "sharp_deg": 70},
    {"far_dist": 1.5, "near_dist": 0.5, "slight_deg": 35, "sharp_deg": 70},
]

test_param_sets(follower_xz, yaw_est_follower, param_list)
