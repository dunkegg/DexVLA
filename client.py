import socket
import struct
import os
import numpy as np
from PIL import Image
from io import BytesIO

# 云端的 IP 和端口（你要根据实际情况替换）
SERVER_IP = '120.48.58.215'
SERVER_PORT = 8888

def send_images(sock, image_paths):
    # --- 1. 发送图片数量 ---
    num_images = len(image_paths)
    sock.sendall(struct.pack('!I', num_images))

    for path in image_paths:
        with open(path, 'rb') as f:
            img_bytes = f.read()
        img_len = len(img_bytes)
        # --- 2. 发送每张图片长度 ---
        sock.sendall(struct.pack('!I', img_len))
        # --- 3. 发送图片内容 ---
        sock.sendall(img_bytes)

    print(f"[Client] Sent {num_images} images")

def recv_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise ConnectionError("Socket closed before receiving all data")
        data += more
    return data

def recv_array(sock):
    # --- 1. 接收 shape ---
    shape_data = recv_all(sock, 8)
    rows, cols = struct.unpack('!II', shape_data)

    # --- 2. 接收 dtype ---
    dtype_len_data = recv_all(sock, 4)
    dtype_len = struct.unpack('!I', dtype_len_data)[0]
    dtype_str = recv_all(sock, dtype_len).decode()

    # --- 3. 接收数据长度 + 内容 ---
    data_len_data = recv_all(sock, 4)
    data_len = struct.unpack('!I', data_len_data)[0]
    data = recv_all(sock, data_len)

    # --- 4. 反序列化 ---
    array = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape((rows, cols))
    print(f"[Client] Received array of shape: {array.shape}")
    return array

def main():
    # 示例图片路径（本地已有的多张图片）
    image_dir = './test_images'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_paths:
        print("No images found to send.")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"[Client] Connected to server {SERVER_IP}:{SERVER_PORT}")

        while True:
            send_images(sock, image_paths)
            result_array = recv_array(sock)

            # 可以在这里处理 result_array，例如保存、打印等
            print("[Client] Result from server:", result_array)

            # 等待下次发送（可加 sleep 或某种触发条件）
            input("Press Enter to send again, or Ctrl+C to exit...")

if __name__ == '__main__':
    main()
