import sys
import os
import random
import numpy as np
import h5py
import cv2
import json
import magnum as mn
from tqdm import tqdm

import argparse
import imageio
from habitat_for_sim.utils.goat import read_yaml, extract_dict_from_folder, get_current_scene, process_episodes_and_goals, convert_to_scene_objects, find_scene_path, calculate_euclidean_distance

# 将上级目录加入 Python 搜索路径

from evaluate_dexvln.raw_robot import RawRobotEnv, qwen2_vla_policy
from evaluate_dexvln.record import create_log_json, append_log

import socket
import struct
import numpy as np
from io import BytesIO
from PIL import Image

# 参数
HOST = '0.0.0.0'  # 监听所有 IP
PORT = 8888       # 你自定义的端口

def recv_all(sock, length):
    """接收指定长度数据"""
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket closed before receiving all data")
        data += packet
    return data

def handle_client(conn, robot):
    while True:
        # --- 1. 接收图片数量 ---
        raw_num_images = recv_all(conn, 4)
        num_images = struct.unpack('!I', raw_num_images)[0]
        print(f"[Server] Expecting {num_images} images")

        images = []
        for _ in range(num_images):
            # --- 2. 接收单张图片大小 ---
            raw_len = recv_all(conn, 4)
            img_len = struct.unpack('!I', raw_len)[0]

            # --- 3. 接收图像内容 ---
            img_bytes = recv_all(conn, img_len)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            images.append(np.array(img))

        print(f"[Server] Received {len(images)} images")
        ##################################### model process
        robot.set_obs(images, 0 , True)
        actions = robot.eval_bc_raw()
        
        # --- 4. 模拟处理：生成一个 numpy array 返回（例子中是图像尺寸信息）---
        # result = np.array([[img.shape[0], img.shape[1], 3] for img in images], dtype=np.int32)
        result = actions

        # --- 5. 序列化并发送 np.array ---
        array_bytes = result.tobytes()
        array_len = len(array_bytes)

        # 先发送 array shape 和 dtype（必要元数据）
        shape = result.shape
        dtype_str = str(result.dtype)

        # 发送 shape（两个整数）
        conn.sendall(struct.pack('!II', shape[0], shape[1]))
        # 发送 dtype 字符串长度 + 内容
        conn.sendall(struct.pack('!I', len(dtype_str)))
        conn.sendall(dtype_str.encode())
        # 发送数据本体
        conn.sendall(struct.pack('!I', array_len))
        conn.sendall(array_bytes)

        print("[Server] Sent array response, waiting for next...")

def start_server(robot):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"[Server] Connected by {addr}")
            handle_client(conn, robot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True,
                        help='Path to the YAML config file')
    args = parser.parse_args()

    cfg = read_yaml(args.yaml_file_path)
    json_data = cfg.json_file_path
    img_output_dir = cfg.img_output_dir
    video_output_dir = cfg.video_output_dir
    log_path = create_log_json() if cfg.log_path is None else cfg.log_path 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 16
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)#############################
        "model_path": cfg.model_path,
        #############################################################################
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
    }

    # fake env for debug
    policy = qwen2_vla_policy(policy_config)
    agilex_bot = RawRobotEnv(policy_config, policy,plot_dir=img_output_dir)
    agilex_bot.reset(10, None)
    start_server(agilex_bot)
    ######################################
    time_step = 0
    timestep_gap = 0.2
    forward_speed = 0.5
    now = timestep_gap * time_step
    last_sample_time = 0
        
    sample_fps = 1.3
    sample_fps = 3
    plan_fps = 10
    follow_size = 5





            

