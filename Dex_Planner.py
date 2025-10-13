#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import io
import tf
import cv2
import time
import math
import rospy
import threading
import numpy as np
import pickle
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import imageio
import torch
from evaluate_dexvln.eval_vln import process_obs, qwen2_vla_policy
from data_utils.utils import set_seed

def angle_mod(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

class PathFinderController:
    def __init__(self, Kp=0.8, Ki=0.0, Kd=0.1,
                 max_v=0.3, max_w=1.0,
                 max_acc_v=0.5, max_acc_w=0.5, dt=0.05):
        """
        单点 PID 控制器
        Kp/Ki/Kd: yaw 方向 PID 参数
        max_v: 最大线速度
        max_w: 最大角速度
        max_acc_v: 最大线加速度
        max_acc_w: 最大角加速度
        dt: 控制周期
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_v = max_v
        self.max_w = max_w
        self.max_acc_v = max_acc_v
        self.max_acc_w = max_acc_w
        self.dt = dt

        self.prev_v = 0.0
        self.prev_w = 0.0
        self.integral_error = 0.0
        self.prev_yaw_error = 0.0

    def calc_control_command(self, x_diff, y_diff, theta, theta_goal):
        """
        x_diff, y_diff: 当前点到目标点的相对坐标
        theta: 当前朝向
        theta_goal: 目标朝向（yaw）
        返回: rho, v, w
        """
        rho = np.hypot(x_diff, y_diff)           # 距离误差
        target_yaw = np.arctan2(y_diff, x_diff)  # 朝向目标点角度
        yaw_error = angle_mod(target_yaw - theta)
        beta = angle_mod(theta_goal - theta - yaw_error)

        # PID 控制角速度
        self.integral_error += yaw_error * self.dt
        derivative = (yaw_error - self.prev_yaw_error) / self.dt
        w_cmd = self.Kp * yaw_error + self.Ki * self.integral_error + self.Kd * derivative
        w_cmd = np.clip(w_cmd, -self.max_w, self.max_w)
        self.prev_yaw_error = yaw_error

        # 如果角度误差大，先旋转
        if abs(yaw_error) > np.pi / 4:
            v_cmd = 0.0
        else:
            v_cmd = min(self.max_v, rho)

        # 线速度加速度限制
        dv = v_cmd - self.prev_v
        dv = np.clip(dv, -self.max_acc_v * self.dt, self.max_acc_v * self.dt)
        v = self.prev_v + dv

        # 角速度加速度限制
        dw = w_cmd - self.prev_w
        dw = np.clip(dw, -self.max_acc_w * self.dt, self.max_acc_w * self.dt)
        w = self.prev_w + dw

        self.prev_v = v
        self.prev_w = w

        return rho, v, w

class DexPlanner:
    def __init__(self, policy_config, save_freq=0.5, max_history=20):
        # load vln
        self.policy_config = policy_config
        self.policy = qwen2_vla_policy(policy_config)

        rospy.init_node("dex_planner", anonymous=True)
        self.MAX_LINEAR_SPEED = 0.5
        self.MAX_ANGULAR_SPEED = 1.0

        # self.bridge = CvBridge()
        self.current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self.history_obs = deque(maxlen=max_history)
        self.obs = None
        self.inference_action = None
        self.history_lock = threading.Lock()
        self.last_save_time = 0.0
        self.save_freq = save_freq

        self.controller = PathFinderController()
        self.current_target = None
        self.target_lock = threading.Lock()
        
        # 相对坐标系管理
        self.reference_frame = None
        self.local_target = None
        self.reference_lock = threading.Lock()
        # 目标点状态管理
        self.linear_x = 0.0
        self.w_angular = 0.0
        self.target_reached = False
        self.need_new_target = True  # 初始需要目标点
        self.video_writer = None
        # ROS订阅发布
        self.image_sub = rospy.Subscriber("/realsense_head/color/image_raw", Image, self.image_callback)
        self.odom_sub = rospy.Subscriber("/Odometry", Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher("/calib_vel", Twist, queue_size=10)
        
        # 启动目标更新线程
        threading.Thread(target=self.target_update_loop, daemon=True).start()

    def ros_image_to_cv2(self, msg: Image) -> np.ndarray:
        """
        将 ROS Image 消息转换为 OpenCV 图像 (numpy array)
        只支持常见编码：'rgb8', 'bgr8', 'mono8'
        """
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            # OpenCV 默认是 BGR，如果需要 BGR 可交换通道
            img = img[:, :, ::-1]  # RGB -> BGR
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
            # cv2.imshow("Image window", cv_image)
            # cv2.waitKey(3)
        except RuntimeError as e:
            rospy.logerr(f"ros_image_to_cv2 Error: {e}")
            return

        self.visualize_trajectory(cv_image, self.inference_action)
        # rospy.loginfo(f"images_len{cv_image.shape}")
        now = time.time()
        if now - self.last_save_time >= self.save_freq:
            # import pdb; pdb.set_trace()
            with self.history_lock:
                self.history_obs.append(cv_image)
                # rospy.loginfo("history2")
                # ret, buf = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                # if ret:
                #     rospy.loginfo("history3")
                #     self.history_obs.append(buf.tobytes())
            self.last_save_time = now

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

        # 初始化 imageio 视频写入器
        if self.video_writer is None:
            os.makedirs("result", exist_ok=True)
            video_path = "result/inference_trajectory.mp4"
            # 使用 imageio 创建视频写入器
            self.video_writer = imageio.get_writer(video_path, fps=10, codec='libx264', quality=8)
            rospy.loginfo(f"[visualize_trajectory] 使用 imageio 视频保存路径: {video_path}")

        # 写入视频帧（10Hz）- imageio 需要 RGB 格式
        try:
            # 将 BGR 转换为 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.video_writer.append_data(img_rgb)
        except Exception as e:
            rospy.logerr(f"[visualize_trajectory] 写入视频帧失败: {e}")


    def odom_callback(self, msg):
        """里程计回调函数"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_position = {'x': x, 'y': y, 'z': z, 'yaw': yaw}
    
    def global_to_local(self, global_point, reference_frame):
        """全局坐标转局部坐标"""
        if global_point is None or reference_frame is None:
            return None
            
        global_x, global_y, global_z, global_yaw = global_point
        ref_x, ref_y, ref_z, ref_yaw = reference_frame['x'], reference_frame['y'], reference_frame['z'], reference_frame['yaw']
        
        dx = global_x - ref_x
        dy = global_y - ref_y
        dz = global_z - ref_z
        
        local_x = dx * math.cos(ref_yaw) + dy * math.sin(ref_yaw)
        local_y = -dx * math.sin(ref_yaw) + dy * math.cos(ref_yaw)
        local_z = dz
        local_yaw = angle_mod(global_yaw - ref_yaw)
        
        return [local_x, local_y, local_z, local_yaw]
    
    def local_to_global(self, local_point, reference_frame):
        """局部坐标转全局坐标"""
        if local_point is None or reference_frame is None:
            return None
            
        local_x, local_y, local_yaw = local_point
        #
        local_x, local_y = local_y, -local_x
        #

        ref_x, ref_y, ref_z, ref_yaw = reference_frame['x'], reference_frame['y'], reference_frame['z'], reference_frame['yaw']
        
        global_x = ref_x + local_x * math.cos(ref_yaw) - local_y * math.sin(ref_yaw)
        global_y = ref_y + local_x * math.sin(ref_yaw) + local_y * math.cos(ref_yaw)
        global_z = ref_z
        global_yaw = angle_mod(ref_yaw + local_yaw)
        
        return [global_x, global_y, global_z, global_yaw]

    def get_obs(self, n_frames=10):
        """获取图像帧"""
        with self.history_lock:
            if not self.history_obs:
                return []
            if len(self.history_obs) < n_frames:
                padding = [self.history_obs[0]] * (n_frames - len(self.history_obs))
                return padding + list(self.history_obs)
            return list(self.history_obs)[-n_frames:]

    def goto_pose(self, goal):
        """基于相对坐标系的导航控制"""
        if goal is None:
            self.stop()
            return

        with self.reference_lock:
            if self.reference_frame is None or self.local_target is None:
                self.stop()
                return
            
            current_local = self.global_to_local(
                [self.current_position['x'], self.current_position['y'], 
                self.current_position['z'], self.current_position['yaw']],
                self.reference_frame
            )
            
            if current_local is None:
                self.stop()
                return
            
            x_diff = self.current_target[0] - current_local[0]
            y_diff = self.current_target[1] - current_local[1]
            theta = current_local[3]
            theta_goal = self.current_target[2]
            # rospy.loginfo(f"x_diff: {x_diff}, y_diff {y_diff}, theta {current_local[3]}, theta_goal{self.current_target[2]}")
        rho, v, w = self.controller.calc_control_command(x_diff, y_diff, theta, theta_goal)

        # 当距离过小停止
        if rho < 0.05:
            v = 0.0
            w = 0.0

        self.linear_x = v
        self.w_angular = w

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

        rospy.loginfo_throttle(1.0, f"[Control] rho={rho:.3f}, alpha={angle_mod(np.arctan2(y_diff, x_diff) - theta):.3f}, "
                                    f"beta={angle_mod(theta_goal - theta - angle_mod(np.arctan2(y_diff, x_diff) - theta)):.3f}, "
                                    f"v={v:.3f}, w={w:.3f}")


    def inference_new_target(self, observations):

        rospy.loginfo(f"inference:  obs len: {len(observations)}")
        raw_lang = "follow the human"
        raw_lang = f"Your task is: {raw_lang}. You are given a sequence of historical visual observations in temporal order (earliest first, latest last). Based on this sequence, predict your future movement trajectory."
        
        """推理新目标点（模拟推理过程）"""
        # assert raw_lang is not None, "raw lang is None!!!!!!"
        set_seed(0)
        rand_crop_resize = False

        self.policy.policy.eval()

        ## 4. load data stats(min,max,mean....) and define post_process####################################
        stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        if policy_config["action_head"].lower() == 'act':
            post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        elif 'scale_dp_policy' in policy_config["action_head"]:
            post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        #############################################################################################################
        compressed = False
        images = []
        # cv2.imshow("Image window", observations[-1])
        for img in observations:

            if compressed:
                img = cv2.imdecode(img, 1)
            # cv2.imshow("Image window", img)
            # cv2.waitKey(3)
            img = cv2.resize(img, (320,240))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = img.astype(np.float32) /255.0
            # cv2.imshow("Image window", img)
            # cv2.waitKey(3)
            # plt.savefig
            images.append(img)

        init_state = np.array([0,0,0])
        obs = {
            'top': images,
        }
        # rospy.loginfo(f"img np: {images[-1]}")
        with torch.inference_mode():
            traj_rgb_np, robot_state = process_obs(obs, init_state, stats)
            robot_state = torch.from_numpy(robot_state).float().cuda()
                ### 6. Augment the images##############################################################################################
            curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
            
            batch = self.policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
            if policy_config['tinyvla']:
                all_actions, outputs = self.policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=self.policy.tokenizer)
            else:
                # from inspect import signature
                # print(signature(policy.policy.generate))
                all_actions, outputs = self.policy.policy.evaluate(**batch, is_eval=True, tokenizer=self.policy.tokenizer)

                all_actions = all_actions.squeeze(0)  #
                all_actions = all_actions.to(dtype=torch.float32).cpu().numpy()
                all_actions = np.array([post_process(raw_action) for raw_action in all_actions])
            
            self.inference_action = all_actions

            return all_actions[-1]
            # timestamp = time.time_ns() // 1_000_000
            # rospy.loginfo(f"traj: {all_actions}")
            # cv2.imshow("Image window", images[-1])
            # plt.imshow(cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB))
            # self.visualize_data(all_actions, time.time() ,images)
            # return [0.0, 0.0, 0.0, 0.0]  # [x, y, z, yaw]

    def target_update_loop(self):
        cur_time = time.time()
        last_plan_time = time.time()
        plan_gap  = 1
        while not rospy.is_shutdown():
            cur_time = time.time()
            self.obs = self.get_obs(n_frames=10)
            if self.obs and cur_time - last_plan_time > plan_gap:
                local_target = self.inference_new_target(self.obs)

                with self.reference_lock:
                    self.reference_frame = self.current_position.copy()  # 使用当前里程计作为参考

                if local_target is not None:
                    with self.reference_lock:
                        self.local_target = local_target
                    
                    global_target = self.local_to_global(local_target, self.reference_frame)
                    
                    with self.target_lock:
                        self.current_target = global_target
                        self.save_obs_traj(self.inference_action, self.obs)
                        rospy.loginfo(f"model point {self.inference_action[-1]}")
                        rospy.loginfo(f"pid point {self.current_target}")
                last_plan_time = cur_time


    def stop(self):
        """停止机器人"""
        twist = Twist()
        self.cmd_pub.publish(twist)

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

    def run(self):
        """主控制循环"""
        rate = rospy.Rate(20) 
        
        rospy.loginfo("[System] 路径规划器启动，等待初始目标点...")
        
        while not rospy.is_shutdown():
            
            with self.target_lock:
                target = self.current_target
                # rospy.loginfo(f"target:{target}")
                # self.visualize_data(target, self.obs)
        
            self.goto_pose(target)
            
            rate.sleep()

if __name__ == "__main__":
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 30
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        "model_path": "OUTPUT/qwen2_follow_real/checkpoint-4000",
        # "model_path": "/vln_ws/DexVLA/OUTPUT/single_follow_normal/checkpoint-20000",
        #############################################################################
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
    }
    planner = DexPlanner(policy_config = policy_config, save_freq=0.5, max_history=20)
    planner.run()