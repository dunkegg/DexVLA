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

import torch
from evaluate_dexvln.eval_vln import process_obs, qwen2_vla_policy
from data_utils.utils import set_seed

def angle_mod(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

class PathFinderController:
    def __init__(self, Kp_rho=0.8, Kp_alpha=1.0, Kp_beta=-0.3):
        self.Kp_rho = Kp_rho
        self.Kp_alpha = Kp_alpha
        self.Kp_beta = Kp_beta

    def calc_control_command(self, x_diff, y_diff, theta, theta_goal):
        rho = np.hypot(x_diff, y_diff)
        v = self.Kp_rho * rho
        alpha = angle_mod(np.arctan2(y_diff, x_diff) - theta)
        beta = angle_mod(theta_goal - theta - alpha)

        if alpha > np.pi / 2 or alpha < -np.pi / 2:
            alpha = angle_mod(np.arctan2(-y_diff, -x_diff) - theta)
            beta = angle_mod(theta_goal - theta - alpha)
            v = -v

        w = self.Kp_alpha * alpha + self.Kp_beta * beta
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
        self.target_reached = False
        self.need_new_target = True  # 初始需要目标点

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
            
        local_x, local_y, local_z, local_yaw = local_point
        ref_x, ref_y, ref_z, ref_yaw = reference_frame['x'], reference_frame['y'], reference_frame['z'], reference_frame['yaw']
        
        global_x = ref_x + local_x * math.cos(ref_yaw) - local_y * math.sin(ref_yaw)
        global_y = ref_y + local_x * math.sin(ref_yaw) + local_y * math.cos(ref_yaw)
        global_z = ref_z + local_z
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
            
            x_diff = self.local_target[0] - current_local[0]
            y_diff = self.local_target[1] - current_local[1]
            theta = current_local[3]
            theta_goal = self.local_target[3]

        rho, v, w = self.controller.calc_control_command(x_diff, y_diff, theta, theta_goal)
        v = np.clip(v, -self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED)
        w = np.clip(w, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)
        
        # 记录控制状态（限流输出）
        rospy.loginfo_throttle(2.0, 
            f"[Control] 局部误差: dist={rho:.3f}, v={v:.3f}, w={w:.3f}")

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
            cv2.imshow("Image window", img)
            cv2.waitKey(3)
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

            # timestamp = time.time_ns() // 1_000_000
            # rospy.loginfo(f"traj: {all_actions}")
            # cv2.imshow("Image window", images[-1])
            # plt.imshow(cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB))
            self.visualize_data(all_actions, time.time() ,images)
            return [0.0, 0.0, 0.0, 0.0]  # [x, y, z, yaw]

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
            

                last_plan_time = cur_time


    def stop(self):
        """停止机器人"""
        twist = Twist()
        self.cmd_pub.publish(twist)

    def visualize_data(self,local_traj, timestamp, history_obs=None):
        """
            local_traj: [x,y, yaw] - list of arrays or numpy array
            robot_pos: [x, y, z] - numpy array
            robot_yaw: yaw - scalar or array
        """
        """可视化局部轨迹、RGB图像并保存"""
        # 创建图形，调整为2行1列布局
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 1, figure=fig)  # 变为2行1列布局
        
        # 处理 local_traj - 转换为 numpy array
        if isinstance(local_traj, list):
            if len(local_traj) > 0 and isinstance(local_traj[0], np.ndarray):
                local_traj = np.array(local_traj)
            else:
                local_traj = np.array(local_traj)
        
        # 1. 绘制RGB图像序列（修正部分）
        ax1 = fig.add_subplot(gs[0, 0])
        if history_obs and len(history_obs) > 0:
            frames_to_show = history_obs[-10:] if len(history_obs) >= 10 else history_obs
            n_frames = len(frames_to_show)
            
            n_cols = min(5, n_frames)
            n_rows = math.ceil(n_frames / n_cols)
            inner_gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[0, 0])
            
            # 按从新到旧顺序显示（最新帧在最前面）
            for i, img_bytes in enumerate(reversed(frames_to_show)):  # 注意：reversed保证新帧在前
                ax = fig.add_subplot(inner_gs[i])
                try:
                    img = Image.open(io.BytesIO(img_bytes))  # 现在 io 模块已导入
                    ax.imshow(np.array(img))  # 显式转换为NumPy数组
                    ax.set_title(f"Frame {n_frames - i}")  # 帧编号从新到旧
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax.axis('off')
        else:
            ax1.text(0.5, 0.5, "No image data", ha='center', va='center')
        ax1.set_title("Recent Camera Frames (Newest to Oldest)")
        ax1.axis('off')
        
        # 2. 绘制局部轨迹 - 显示机器人朝向和轨迹相对位置
        ax2 = fig.add_subplot(gs[1, 0])
        if hasattr(local_traj, 'size') and local_traj.size > 0:
            x = local_traj[:, 0]
            y = local_traj[:, 1]
            
            # 绘制轨迹
            ax2.plot(x, y, 'b-', linewidth=2, label='Local trajectory')
            ax2.plot(x[0], y[0], 'go', markersize=8, label='Start')
            ax2.plot(x[-1], y[-1], 'mo', markersize=8, label='End')
            
            # 绘制机器人位置（局部坐标系原点）
            ax2.plot(0, 0, 'ro', markersize=10, label='Robot')
            
            # 绘制机器人朝向箭头
            robot_direction_x = 0.3 * np.cos(0)  # 机器人朝向角为0（局部坐标系）
            robot_direction_y = 0.3 * np.sin(0)
            ax2.arrow(0, 0, robot_direction_x, robot_direction_y, 
                    head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
            
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
        
        # 添加时间戳
        fig.suptitle(f"Time: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}", y=0.02)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join("sample", f"traj_vis_{int(timestamp*1000)}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

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