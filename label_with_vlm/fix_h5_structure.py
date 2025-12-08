import h5py
import os
import sys

def fix_h5_prefix(h5_path):
    """
    修复 HDF5 文件中冗余的 /annotations_status/annotations_status 嵌套结构。
    将所有数据从第二层移动到第一层，然后删除空的第二层群组。
    """
    
    # 检查文件是否存在
    if not os.path.exists(h5_path):
        # 批量处理时，这里通常不会被触发，但保持检查是好习惯
        print(f"❌ 错误：文件未找到：{h5_path}") 
        return

    # 定义冗余结构路径
    OUTER_GROUP = "annotations_status"
    INNER_GROUP = "annotations_status"
    INNER_PATH = f"/{OUTER_GROUP}/{INNER_GROUP}"

    try:
        # 使用 'r+' 模式打开文件，允许读写
        with h5py.File(h5_path, 'r+') as f:
            print(f"\n🔍 正在检查文件: {os.path.basename(h5_path)}") # 打印文件名，更简洁

            # 1. 检查冗余结构是否存在
            if OUTER_GROUP in f and INNER_GROUP in f[OUTER_GROUP]:
                
                inner_group = f[INNER_PATH]
                items_to_move = list(inner_group.keys()) # 找到要移动的实际数据集（如 status_0, status_1）
                
                if not items_to_move:
                    print(f"⚠️ 发现冗余结构 {INNER_PATH}，但内部是空的，正在删除...")
                    del f[INNER_PATH]
                    return

                print(f"✅ 发现冗余结构: {INNER_PATH}。准备移动 {len(items_to_move)} 个数据集...")
                
                for item_name in items_to_move:
                    # 完整的源路径：/annotations_status/annotations_status/status_0
                    source_path = f"{INNER_PATH}/{item_name}"
                    # 目标路径：/annotations_status/status_0
                    dest_path = f"/{OUTER_GROUP}/{item_name}"
                    
                    # 使用 f.move() 进行移动（相当于重命名到新路径）
                    f.move(source_path, dest_path)
                    print(f"   -> 移动成功: {item_name}")
                
                # 2. 删除现在为空的冗余内部群组
                del f[INNER_PATH]
                print(f"🗑️ 已删除空的冗余群组: {INNER_PATH}")
                print("🎉 HDF5 文件结构修复完毕！")
            
            else:
                print("👍 未发现预期的冗余结构，无需修复。")
                
    except Exception as e:
        print(f"❌ 修复过程中发生错误: {e}")
        # 打印正在处理的文件名，方便排查是哪个文件出问题
        print(f"--- 错误发生在文件: {h5_path} ---")
        # 这里的 return 可能会跳过后续文件，如果希望继续处理，可以改成 pass
        # pass
        
        
if __name__ == "__main__":
    # >>>>>> 根目录路径 <<<<<<
    # 请确保这是 rxr_smooth 文件夹的绝对路径
    BASE_DIR = "/mnt/pfs/3zpd5q/code/eval/DexVLA/data/raw_data/rxr_smooth" 
    
    if not os.path.isdir(BASE_DIR):
        print(f"❌ 错误：指定的路径不是一个有效的目录：{BASE_DIR}")
        sys.exit(1)

    print(f"🚀 开始批量处理目录: {BASE_DIR}")
    
    # 遍历目录下的所有文件
    for filename in os.listdir(BASE_DIR):
        if filename.endswith(".hdf5"):
            full_path = os.path.join(BASE_DIR, filename)
            # 对每个找到的 HDF5 文件调用修复函数
            fix_h5_prefix(full_path)
            
    print("\n\n🎉🎉🎉 所有 HDF5 文件检查及修复完成！")