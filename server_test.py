import socket
import struct
import numpy as np
from io import BytesIO
from PIL import Image

HOST = '0.0.0.0'
PORT = 8888

def recv_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise ConnectionError("Connection closed unexpectedly")
        data += more
    return data

def handle_client(conn):
    while True:
        raw_num_images = recv_all(conn, 4)
        num_images = struct.unpack('!I', raw_num_images)[0]
        print(f"[Server] Expecting {num_images} images")

        images = []
        for _ in range(num_images):
            raw_len = recv_all(conn, 4)
            img_len = struct.unpack('!I', raw_len)[0]
            img_bytes = recv_all(conn, img_len)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            images.append(np.array(img))

        print(f"[Server] Received {len(images)} images")
        print(images)
        # 模拟返回：给客户端返回图片高宽数组
        result = np.array([[img.shape[0], img.shape[1], 3] for img in images], dtype=np.int32)
        shape = result.shape
        dtype_str = str(result.dtype)
        array_bytes = result.tobytes()
        array_len = len(array_bytes)

        # 发送形状和dtype等元数据
        conn.sendall(struct.pack('!II', shape[0], shape[1]))
        conn.sendall(struct.pack('!I', len(dtype_str)))
        conn.sendall(dtype_str.encode())
        conn.sendall(struct.pack('!I', array_len))
        conn.sendall(array_bytes)

        print("[Server] Sent array response, waiting for next batch...")

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