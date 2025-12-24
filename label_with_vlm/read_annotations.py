import h5py

# 指定你的文件路径
h5_path = "data/raw_data/rxr_smooth/episode_999.hdf5"

# 打开文件并读取
with h5py.File(h5_path, "r") as f:
    print("Keys in this HDF5 file:", list(f.keys()))

    # 读取 annotations_status1
    if "annotations_status0" in f:
        # status_labels = f["annotations_status"]["status_1"][:]
        status_labels = f["annotations_status0"][:]
        print(f"\n✅ 共 {len(status_labels)} 条 status 标签:\n")
        for i, ann in enumerate(status_labels):
            print(f"[{i:03d}] 状态: {ann.decode('utf-8') if isinstance(ann, bytes) else ann}")
    else:
        print("⚠️ 没有找到 'annotations_status' 键")

# import h5py

# # 指定你的文件路径
# h5_path = "data/raw_data/rxr_smooth/episode_1234.hdf5"

# # 打开文件并读取
# with h5py.File(h5_path, "r") as f:
#     print("Keys in this HDF5 file:", list(f.keys()))

#     if "annotations_status" in f:
#         status_group_1 = f["annotations_status"]
        
#         # 打印第二层键名 (你之前已经确认是 ['annotations_status'])
#         print(f"\n🔑 'annotations_status' (Level 1) 内部键: {list(status_group_1.keys())}")
        
#         # --- 关键修改：检查第三层结构 ---
#         if "annotations_status" in status_group_1:
#             status_group_2 = status_group_1["annotations_status"]
#             # 打印第三层键名
#             print(f"🔑 'annotations_status' (Level 2) 内部键: {list(status_group_2.keys())}")
#         # --- 关键修改结束 ---

#         # 原始错误代码，保持注释或删除
#         # status_labels = f["annotations_status"]["annotations_status"][:] 
    
#     else:
#         print("⚠️ 没有找到 'annotations_status' 键")