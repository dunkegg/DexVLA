import sys
import os
import random
import numpy as np
import h5py
import cv2
import json
# import magnum as mn
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

def handle_client(conn, addr, robot):
    print(f"[Server] Connected by {addr}")
    try:
        while True:
            # --- 1. 接收图片数量 ---
            raw_num_images = recv_all(conn, 4)
            num_images = struct.unpack('!I', raw_num_images)[0]
            print(f"[Server] Expecting {num_images} images")

            images = []
            for _ in range(num_images):
                raw_len = recv_all(conn, 4)
                img_len = struct.unpack('!I', raw_len)[0]
                img_bytes = recv_all(conn, img_len)
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (320, 240))
                images.append(img)

            print(f"[Server] Received {len(images)} images")

            # 模型推理
            robot.set_obs(images, 0, True)
            actions = robot.eval_bc_raw()

            # 序列化返回
            result = actions
            array_bytes = result.tobytes()
            array_len = len(array_bytes)
            shape = result.shape
            dtype_str = str(result.dtype)

            conn.sendall(struct.pack('!II', shape[0], shape[1]))
            conn.sendall(struct.pack('!I', len(dtype_str)))
            conn.sendall(dtype_str.encode())
            conn.sendall(struct.pack('!I', array_len))
            conn.sendall(array_bytes)

            print("[Server] Sent array response, waiting for next...")

    except (ConnectionError, OSError) as e:
        print(f"[Server] Client {addr} disconnected: {e}")
    finally:
        conn.close()
        print(f"[Server] Connection with {addr} closed.")

def start_server(robot):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            handle_client(conn, addr, robot)
            # 一旦 client 断开，这里会自动返回，重新等待下一次连接


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





            

