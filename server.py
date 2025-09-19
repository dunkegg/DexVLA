
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

def handle_client(conn):
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

        # --- 4. 模拟处理：生成一个 numpy array 返回（例子中是图像尺寸信息）---
        result = np.array([[img.shape[0], img.shape[1], 3] for img in images], dtype=np.int32)

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

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"[Server] Connected by {addr}")
            handle_client(conn)

if __name__ == '__main__':
    start_server()
