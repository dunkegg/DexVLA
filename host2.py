# client_udp_stream.py
import socket
import numpy as np
import cv2
import threading

# UDP 配置
HOST_IP = "120.48.58.215"  # 主机IP
UDP_PORT_IMG = 10808  # 接收图像的端口
UDP_PORT_PATH = 8892  # 发送路径的端口

# 创建 UDP Socket
udp_img_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 收图像
udp_img_receiver.bind(("0.0.0.0", UDP_PORT_IMG))
udp_path_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    # 发路径

def receive_image_stream():
    """接收图像数据流但不显示"""
    while True:
        data, _ = udp_img_receiver.recvfrom(65535)  # UDP 单包最大约 64KB
        # 解码 JPEG 字节流（不显示）
        img_np = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 可在此处处理图像数据（如保存或转发）
        print(f"Received image: shape={img_np.shape}")  # 仅打印图像尺寸

def send_path_stream():
    """发送路径数据流"""
    while True:
        path_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 模拟路径
        udp_path_sender.sendto(path_np.tobytes(), (HOST_IP, UDP_PORT_PATH))

# 多线程并行收发
threading.Thread(target=receive_image_stream, daemon=True).start()
threading.Thread(target=send_path_stream, daemon=True).start()

try:
    while True:
        pass
except KeyboardInterrupt:
    print("Client stopped.")
finally:
    udp_img_receiver.close()
    udp_path_sender.close()