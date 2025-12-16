# import h5py
# import numpy as np
# import os
# from h5py import special_dtype

# rxr_dir = "data/raw_data/rxr_smooth" # 修改为自己的文件路径

# def add_instructions_to_h5(h5_path):
#     print(f"\n🔧 正在处理: {h5_path}")
#     with h5py.File(h5_path, "a") as f:

#         if "obs" not in f or "color_0_0" not in f["obs"]:
#             print("⚠️ 跳过：文件缺少 obs/color_0_0")
#             return

#         num_entries = f["obs"]["color_0_0"].shape[0]
#         print(f"  -> 共 {num_entries} 条数据")

#         # 如果 instructions 存在则删除重建
#         if "instructions" in f:
#             del f["instructions"]

#         str_dt = special_dtype(vlen=str)
#         instructions_data = np.array(["walk"] * num_entries, dtype=object)

#         f.create_dataset("instructions", data=instructions_data, dtype=str_dt)

#         print("  ✅ 已添加 instructions=['walk'] * num_entries")


# # ==============================
# # 批量处理整个文件夹
# # ==============================
# files = sorted([f for f in os.listdir(rxr_dir) if f.endswith(".hdf5")])

# print("📂 发现 HDF5 文件:")
# for f in files:
#     print(" -", f)

# for fname in files:
#     add_instructions_to_h5(os.path.join(rxr_dir, fname))

# print("\n🎉 所有 HDF5 文件已完成 instructions 标注！")
import h5py
import numpy as np
# import os
from h5py import special_dtype

def add_instructions_to_h5(h5_path):
    print(f"\n🔧 正在处理: {h5_path}")
    with h5py.File(h5_path, "a") as f:

        if "obs" not in f :
            print("⚠️ 跳过：文件缺少 obs")
            return

        num_entries = f["obs"].shape[0]
        print(f"  -> 共 {num_entries} 条数据")

        # 如果 instruction 存在则删除重建
        if "instruction" in f:
            del f["instruction"]

        str_dt = special_dtype(vlen=str)
        # str_dt = h5py.string_dtype(encoding='utf-8')
        instructions_data = np.array(["walk"] * num_entries, dtype=object)

        f.create_dataset("instruction", data=instructions_data, dtype=str_dt)

        print("  ✅ 已添加 instruction=['walk'] * num_entries")



'''单个episode'''
# 指定单个 episode 文件路径
h5_file_path = "vln_data_4.hdf5"

print("📂 准备处理单个 HDF5 文件:")
print(" -", h5_file_path)

# 调用你已有的函数
add_instructions_to_h5(h5_file_path)

print("\n🎉 单个 episode HDF5 文件已完成 instruction 标注！")