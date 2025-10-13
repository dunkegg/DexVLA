#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import time
import math
import rospy
import numpy as np
import pickle
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from navi_types.msg import TaskInfo, Waypoint
import torch
import tf
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from evaluate_dexvln.eval_vln import process_obs, qwen2_vla_policy
from data_utils.utils import set_seed

class DexPlanner:
    def __init__(self, policy_config, save_freq=0.5, max_history=20):
        # load vln
        self.policy_config = policy_config
        self.policy = qwen2_vla_policy(policy_config)

        rospy.init_node("dex_planner", anonymous=True)
        self.MAX_LINEAR_SPEED = 0.5
        self.MAX_ANGULAR_SPEED = 1.0

        self.history_obs = deque(maxlen=max_history)  # 存储图像
        self.last_save_time = 0.0
        self.save_freq = save_freq
        self.inference_action = None
        self.linear_x = 0.0
        self.w_angular = 0.0
        # ROS订阅发布
        self.image_sub = rospy.Subscriber("/realsense_head/color/image_raw", Image, self.image_callback)
        self.cmd_sub = rospy.Subscriber("/calib_vel", Twist, self.cmd_callback)
        self.taskinfo_pub = rospy.Publisher("/task_info", TaskInfo, queue_size=10)

    def ros_image_to_cv2(self, msg: Image) -> np.ndarray:
        """ROS Image -> OpenCV 图像"""
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)[:, :, ::-1]
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == 'mono8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        return img

    def image_callback(self, msg):
        """保存图像到历史容器"""
        try:
            cv_image = self.ros_image_to_cv2(msg)
        except RuntimeError as e:
            rospy.logerr(f"ros_image_to_cv2 Error: {e}")
            return
        
        self.visualize_trajectory(cv_image, self.inference_action)

        now = time.time()
        if now - self.last_save_time >= self.save_freq:
            self.history_obs.append(cv_image)
            self.last_save_time = now

    def cmd_callback(self, msg: Twist):
        """订阅机器人线速度和角速度"""
        self.linear_x = msg.linear.x
        self.w_angular = msg.angular.z

    def visualize_trajectory(self, cv_image, all_actions):
        """
        在图像上绘制推理轨迹，并以固定10Hz保存视频帧
        all_actions: np.ndarray of shape (N, 3) -> [x, y, yaw]
        """
        if cv_image is None or all_actions is None:
            return

        # 记录时间控制帧率
        now = time.time()
        if not hasattr(self, "last_video_write_time"):
            self.last_video_write_time = 0.0
        if now - self.last_video_write_time < 0.1:  # 10Hz 写帧
            return
        self.last_video_write_time = now

        # 拷贝原图
        img = cv_image.copy()
        h, w, _ = img.shape

        # 定义图像底部中心为机器人中心
        center_x, center_y = w // 2, h - 1
        scale = 50.0  # 每米多少像素，可调
        
        # 绘制轨迹
        for i in range(len(all_actions) - 1):
            x1, y1 = all_actions[i, :2]
            x2, y2 = all_actions[i + 1, :2]
            p1 = (int(center_x + x1 * scale), int(center_y - y1 * scale))
            p2 = (int(center_x + x2 * scale), int(center_y - y2 * scale))
            cv2.line(img, p1, p2, (0, 255, 0), 2)

        # 绘制机器人中心点
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

        text = f"Linear: {self.linear_x:.2f} m/s | Angular: {self.w_angular:.2f} rad/s"
        cv2.rectangle(img, (10, 10), (450, 50), (0, 0, 0), -1)
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 初始化视频写入器
        if not hasattr(self, "video_writer") or self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs("result", exist_ok=True)
            video_path = "result/inference_trajectory.mp4"
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
            rospy.loginfo(f"[visualize_trajectory] 视频保存路径: {video_path}")

        # 写入视频帧（10Hz）
        self.video_writer.write(img)

        # 可选：实时预览
        # cv2.imshow("Trajectory Visualization", img)
        # cv2.waitKey(1)


    
    def get_obs(self, n_frames=10):
        """获取历史图像序列"""
        if not self.history_obs:
            return []
        if len(self.history_obs) < n_frames:
            padding = [self.history_obs[0]] * (n_frames - len(self.history_obs))
            return padding + list(self.history_obs)
        return list(self.history_obs)[-n_frames:]

    def inference_new_target(self, observations):
        """根据观测序列推理路径"""
        rospy.loginfo(f"inference: obs len: {len(observations)}")
        raw_lang = "follow the human"
        raw_lang = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."
        
        set_seed(0)
        self.policy.policy.eval()

        stats_path = os.path.join("/".join(self.policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        if self.policy_config["action_head"].lower() == 'act':
            post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        elif 'scale_dp_policy' in self.policy_config["action_head"]:
            post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

        images = []
        for img in observations:
            img = cv2.resize(img, (320, 240))
            images.append(img)

        init_state = np.array([0,0,0])
        obs = {'top': images}

        with torch.inference_mode():
            traj_rgb_np, robot_state = process_obs(obs, init_state, stats)
            robot_state = torch.from_numpy(robot_state).float().cuda()
            curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
            
            batch = self.policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
            if self.policy_config['tinyvla']:
                all_actions, outputs = self.policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=self.policy.tokenizer)
            else:
                all_actions, outputs = self.policy.policy.evaluate(**batch, is_eval=True, tokenizer=self.policy.tokenizer)
                
                all_actions = all_actions.squeeze(0)  #
                all_actions = all_actions.to(dtype=torch.float32).cpu().numpy()
                all_actions = np.array([post_process(raw_action) for raw_action in all_actions])
            
            self.inference_action = all_actions
            
            return all_actions[-1] # x y yaw

    def publish_taskinfo(self, waypoints):
        """发布 TaskInfo 消息"""
        if not waypoints:
            return
        taskinfo = TaskInfo()
        taskinfo.header.stamp = rospy.Time.now()
        taskinfo.waypoints = waypoints
        self.taskinfo_pub.publish(taskinfo)
        rospy.loginfo(f"[TaskInfo] 发布目标点: {[ (wp.pose.position.x, wp.pose.position.y) for wp in waypoints ]}")

    def run(self):
        """主循环"""
        rate = rospy.Rate(20)  # 20Hz控制频率
        rospy.loginfo("[System] 路径规划器启动，等待目标点...")

        while not rospy.is_shutdown():
            obs = self.get_obs(10)
            if not obs:  # 如果没有任何图像，直接跳过
                rate.sleep()
                continue

            # 推理路径
            path = self.inference_new_target(obs) # x y yaw
            path = np.array(path, dtype=np.float32).reshape(-1, 3)
            # 保存推理后的图像
            self.save_obs_traj(self.inference_action, obs)
            # 创建Waypoints并发布 y-> x -x-> y
            waypoints = []
            for i, pt in enumerate(path):
                waypoint = Waypoint()
                waypoint.id = i + 1
                waypoint.pose.position.x = pt[1]
                waypoint.pose.position.y = -pt[0]
                waypoint.pose.position.z = 0.0
                quat = tf.transformations.quaternion_from_euler(0.0, 0.0, pt[2])
                waypoint.pose.orientation.x = quat[0]
                waypoint.pose.orientation.y = quat[1]
                waypoint.pose.orientation.z = quat[2]
                waypoint.pose.orientation.w = quat[3]
                waypoints.append(waypoint)
            
            self.publish_taskinfo(waypoints)

            rate.sleep()

    def save_obs_traj(self, local_traj, history_obs):
        """可视化局部轨迹和RGB图像并保存到result文件夹"""
        if not os.path.exists("result"):
            os.makedirs("result")

        timestamp = time.time()
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 1, figure=fig)

        # 1. 绘制RGB图像序列
        ax1 = fig.add_subplot(gs[0, 0])
        if history_obs and len(history_obs) > 0:
            frames_to_show = history_obs[-10:] if len(history_obs) >= 10 else history_obs
            n_frames = len(frames_to_show)
            n_cols = min(5, n_frames)
            n_rows = (n_frames + n_cols - 1) // n_cols
            inner_gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[0, 0])
            for i, img in enumerate(reversed(frames_to_show)):
                ax = fig.add_subplot(inner_gs[i])
                ax.imshow(img[..., ::-1] if img.ndim == 3 else img, aspect='auto')
                ax.set_title(f"Frame {n_frames - i}")
                ax.axis('off')
        else:
            ax1.text(0.5, 0.5, "No image data", ha='center', va='center')
        ax1.set_title("Recent Camera Frames (Newest to Oldest)")
        ax1.axis('off')

        # 2. 绘制局部轨迹
        ax2 = fig.add_subplot(gs[1, 0])
        if local_traj is not None and len(local_traj) > 0:
            local_traj = np.array(local_traj)
            x, y = local_traj[:, 0], local_traj[:, 1]
            ax2.plot(x, y, 'b-', linewidth=2, label='Local trajectory')
            ax2.plot(x[0], y[0], 'go', markersize=8, label='Start')
            ax2.plot(x[-1], y[-1], 'mo', markersize=8, label='End')
            ax2.plot(0, 0, 'ro', markersize=10, label='Robot')
            ax2.arrow(0, 0, 0.3*np.cos(np.pi/2), 0.3*np.sin(np.pi/2),head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
            ax2.text(-0.9, 0.9,
                     f"Linear: {self.linear_x:.2f} m/s\nAngular: {self.w_angular:.2f} rad/s",
                     fontsize=12, color='orange', bbox=dict(facecolor='black', alpha=0.5))
            ax2.set_xlim(-1.0, 1.0)
            ax2.set_ylim(-1.0, 1.0)
            ax2.set_title("Local Trajectory with Orientation")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.legend()
            ax2.grid(True)
            ax2.axis('equal')
        else:
            ax2.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
            ax2.set_title("Local Trajectory with Orientation")

        fig.suptitle(f"Time: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}", y=0.02)
        plt.tight_layout()
        save_path = os.path.join("result", f"traj_vis_{int(timestamp*1000)}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        rospy.loginfo(f"[Visualize] Saved trajectory visualization to {save_path}")

if __name__ == "__main__":
    action_head = 'scale_dp_policy'
    policy_config = {
        "model_path": "OUTPUT/qwen2_follow_real/checkpoint-4000",
        "model_base": None,
        "enable_lora": False,
        "action_head": action_head,
        "tinyvla": False,
    }
    planner = DexPlanner(policy_config=policy_config, save_freq=0.5, max_history=20)
    planner.run()