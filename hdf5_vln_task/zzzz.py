import os
import shutil
import random

src_dir = 'data/vln'
dst_dir = 'data/test2'
os.makedirs(dst_dir, exist_ok=True)

files = sorted(os.listdir(src_dir))[:5000]
files = random.sample(files, k=200)
for fname in files:
    shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
