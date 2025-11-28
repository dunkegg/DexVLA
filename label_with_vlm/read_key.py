import h5py

# 要修改的 .h5 文件路径
h5_path = "data/raw_data/rxr_smooth/episode_0.hdf5"
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

# import h5py

# # 指定你的文件路径
# h5_path = "data/raw_data/rxr_smooth/episode_0.hdf5"

# # 打开文件并读取
# with h5py.File(h5_path, "r") as f:
#     print("Keys in this HDF5 file:", list(f.keys()))
#     print(type(f["annotations_status"]))
#     print(list(f["annotations_status"].keys()))
#     # 读取 annotations_status
#     if "annotations_status" in f:
#         status_labels = f["annotations_status"][:]
#         print(f"\n✅ 共 {len(status_labels)} 条 status 标签:\n")
#         for i, ann in enumerate(status_labels):
#             print(f"[{i:03d}] 状态: {ann.decode('utf-8') if isinstance(ann, bytes) else ann}")
#     else:
#         print("⚠️ 没有找到 'annotations_status' 键")

#     # 读取 annotations_action1
#     if "annotations_actions1" in f:
#         action_labels = f["annotations_actions1"][:]
#         print(f"\n✅ 共 {len(action_labels)} 条 action 标签:\n")
#         for i, ann in enumerate(action_labels):
#             print(f"[{i:03d}] 动作: {ann.decode('utf-8') if isinstance(ann, bytes) else ann}")
#     else:
#         print("⚠️ 没有找到 'annotations_actions1' 键")

