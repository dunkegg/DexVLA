
# from PIL import Image
# from transformers import AutoModel, AutoConfig
# from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
# import torch
# import torchvision.transforms as T
# from torchvision.transforms import InterpolationMode

image_path = "/mnt/pfs/3zpd5q/code/EVA-CLIP-8B/CLIP.png"
# image_path = "/mnt/pfs/s7fsio/code/EVA-CLIP-8B/CLIP.png"

captions = ["a diagram", "a dog", "a cat"]

import requests

resp = requests.post(
    "http://localhost:8000/clip",
    files={"image": open(image_path, "rb")},
    data={"texts": captions},
    proxies={"http": None, "https": None}
)
print(resp.status_code)
print(resp.text)  # 看看服务器返回了什么
print(f"Label probs: {resp.json()}")