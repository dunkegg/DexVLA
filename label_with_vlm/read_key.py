import h5py

# 要修改的 .h5 文件路径
h5_path = "label_with_vlm/hyz_data/episode_8.hdf5"
# h5_path = "/wangzejin/code/DexVLA/hyz_test/datasets/1_img_CAM_A_compressed.h5"

with h5py.File(h5_path, 'a') as f:  # 以可写模式打开
    print("📂 删除前的键(datasets/groups):")
    print("Keys in this HDF5 file:", list(f.keys()))
    print("--------------------------------------------------")

    # # # # 要删除的键列表
    keys_to_delete = ['annotations_status']
    # keys_to_delete = ['annotations_actions1', 'annotations_status']

    for key in keys_to_delete:
        if key in f:
            del f[key]
            print(f"✅ 已删除 '{key}'")
        else:
            print(f"ℹ️ 未找到 '{key}'")

    print("--------------------------------------------------")
    print("📂 删除后的键(datasets/groups):")
    print(list(f.keys()))