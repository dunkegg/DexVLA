import h5py

# 要修改的 .h5 文件路径
h5_path = "data/raw_data/rxr_smooth/episode_10.hdf5"
# h5_path = "data/proc_data/test_real/episode_000.h5"
# h5_path = "/wangzejin/code/DexVLA/hyz_test/datasets/1_img_CAM_A_compressed.h5"

with h5py.File(h5_path, 'a') as f:  # 以可写模式打开
    print("📂 删除前的键(datasets/groups):")
    print("Keys in this HDF5 file:", list(f.keys()))
    print("--------------------------------------------------")

    # # # # # 要删除的键列表
    # keys_to_delete = ['annotations_status']
    # # keys_to_delete = ['annotations_actions1', 'annotations_status']

    # for key in keys_to_delete:
    #     if key in f:
    #         del f[key]
    #         print(f"✅ 已删除 '{key}'")
    #     else:
    #         print(f"ℹ️ 未找到 '{key}'")

    # print("--------------------------------------------------")
    # print("📂 删除后的键(datasets/groups):")
    # print(list(f.keys()))

# with h5py.File(h5_path, "r") as f:
#     print("Keys in root:", list(f.keys()))
    
#     obs_group = f["obs"]
#     print("\n--- obs group keys ---")
#     print(list(obs_group.keys()))

#     # 查看每个子 dataset 的 shape 和 dtype
#     for k in obs_group.keys():
#         item = obs_group[k]
#         print(f"{k}: type={type(item)}")
#         if isinstance(item, h5py.Dataset):
#             print("  shape:", item.shape)
#             print("  dtype:", item.dtype)
with h5py.File(h5_path, "r") as f:
    color_group = f["obs"]["color_0_0"]
    print("Keys in color_0_0:", list(color_group.keys()))

    # 打印每一个项的信息
    for k in color_group.keys():
        item = color_group[k]
        print(f"{k}: type={type(item)}")
        if isinstance(item, h5py.Dataset):
            print("  shape:", item.shape)
            print("  dtype:", item.dtype)