import h5py

def print_h5_structure(h5_path):
    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[Group]    {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"[Dataset]  {name}  shape={obj.shape}  dtype={obj.dtype}")

    with h5py.File(h5_path, "r") as f:
        f.visititems(visitor)

# 使用示例
print_h5_structure("data/raw_data/obj/move_new/0/episode_0.hdf5")
# print_h5_structure("data/proc_data/test_real/isaac_f4_C_episode_000.hdf5")
