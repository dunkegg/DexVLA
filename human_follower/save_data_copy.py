import h5py, math, numpy as np, magnum as mn
import quaternion as qt          # pip install numpy-quaternion
import os
import json

# ---------- 辅助转换 ---------- #
def to_vec3(arr_like):
    """
    任意 Vector3 表示 → np.float32[3]
    支持：Magnum Vector3 / ndarray / list / tuple
    """
    if isinstance(arr_like, mn.Vector3):
        return np.array([arr_like.x, arr_like.y, arr_like.z], dtype=np.float32)
    arr = np.asarray(arr_like, dtype=np.float32).reshape(3)
    return arr

def to_quat(arr_like):
    """
    任意四元数 → np.float32[4]  (w, x, y, z)
    支持：Magnum Quaternion / numpy-quaternion / list / tuple / ndarray
    """
    if isinstance(arr_like, mn.Quaternion):
        return np.array([arr_like.scalar,
                         arr_like.vector.x,
                         arr_like.vector.y,
                         arr_like.vector.z], dtype=np.float32)
    if isinstance(arr_like, qt.quaternion):
        return np.array([arr_like.w, arr_like.x, arr_like.y, arr_like.z],
                        dtype=np.float32)
    arr = np.asarray(arr_like, dtype=np.float32).reshape(4)
    # 若给成 (x,y,z,w) 可自动调整 —— 以最后一个元素绝对值最大视为 w
    if abs(arr[0]) < abs(arr[3]):           # 猜测是 [x,y,z,w]
        arr = arr[[3, 0, 1, 2]]
    return arr.astype(np.float32)

# --------- 通用：把对象递归转成可 JSON 的纯 Python --------- #
def _pyify(obj):
    # 基础类型 & None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # magnum 类型
    if isinstance(obj, mn.Vector3):
        return [float(obj.x), float(obj.y), float(obj.z)]
    if isinstance(obj, mn.Quaternion):
        return [float(obj.scalar), float(obj.vector.x), float(obj.vector.y), float(obj.vector.z)]

    # numpy 系列
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 容器递归
    if isinstance(obj, (list, tuple)):
        return [_pyify(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _pyify(v) for k, v in obj.items()}

    # 其他未知类型，做个保底
    return str(obj)



# --------- 修改点 ①：obs 既支持 list[np.ndarray] 也支持 list[{"color_0_0":ndarray}] --------- #
def save_obs_list(obs_list, h5file, sensor_key="color_0_0"):
    """
    obs_list : List[np.ndarray] 或 List[Dict[str, np.ndarray]]
    保存到 /obs/{sensor_key}
    """
    frames = []
    for ob in obs_list:
        if isinstance(ob, dict):
            frames.append(np.asarray(ob[sensor_key]))
        else:
            frames.append(np.asarray(ob))
    # 尺寸一致则 stack，否则逐帧存
    shapes = {f.shape for f in frames}
    if len(shapes) == 1 and len(frames) > 0:
        dataset = np.stack(frames, 0).astype(np.uint8)
        h5file.create_dataset(f"obs/{sensor_key}", data=dataset, compression="gzip")
    else:
        grp = h5file.create_group(f"obs/{sensor_key}")
        for i, img in enumerate(frames):
            grp.create_dataset(f"{i:06d}", data=img.astype(np.uint8), compression="gzip")

# --------- 修改点 ②：新增 follow_groups_json；修复 shortest_path 覆写错误 --------- #
def save_output_to_h5(output: dict, h5_path="output.h5"):
    """
    兼容两种结构：
      - 旧：output['follow_paths']  每条包含 {'obs_idx','type','desc','follow_state','human_state','path', 'shortest_path'(可选)}
      - 新：output['follow_groups']  多候选结构（我们以 JSON 一次性写入）
    观测写入：/obs/color_0_0
    """
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        # 1) 观测
        save_obs_list(output.get("obs", []), f, sensor_key="color_0_0")

        # 2) 旧格式：/follow_paths/*
        if "follow_paths" in output and len(output["follow_paths"]) > 0:
            grp = f.create_group("follow_paths")
            for k, fp in enumerate(output["follow_paths"]):
                g = grp.create_group(f"{k:06d}")
                g.create_dataset("obs_idx", data=np.int32(fp["obs_idx"]))
                g.create_dataset("type", data=np.int32(fp["type"]))
                dt = h5py.string_dtype(encoding='utf-8')
                g.create_dataset("desc", data=fp.get("desc",""), dtype=dt)

                # follow_state
                fpos, fquat, fyaw = fp["follow_state"]
                g.create_dataset("follow_pos", data=to_vec3(fpos))
                g.create_dataset("follow_quat", data=to_quat(fquat))
                g.create_dataset("follow_yaw",  data=np.float32(fyaw))

                # human_state
                hpos, hquat, hyaw = fp["human_state"]
                g.create_dataset("human_pos", data=to_vec3(hpos))
                g.create_dataset("human_quat", data=to_quat(hquat))
                g.create_dataset("human_yaw",  data=np.float32(hyaw))

                # 主路径 path → Nx8: [x,y,z,w,x,y,z,yaw]
                path_list = fp["path"]
                path_np = np.empty((len(path_list), 8), np.float32)
                for i, (pos, quat, yaw) in enumerate(path_list):
                    path_np[i, :3]  = to_vec3(pos)
                    path_np[i, 3:7] = to_quat(quat)
                    path_np[i, 7]   = np.float32(yaw)
                g.create_dataset("rel_path", data=path_np, compression="gzip")

                # shortest_path（如果有），注意不能复用 path_np！
                if "shortest_path" in fp and fp["shortest_path"] is not None:
                    sp_list = fp["shortest_path"]  # 约定为 [ (pos,quat,yaw) ] 或 [pos]，按你实际
                    if len(sp_list) > 0:
                        if isinstance(sp_list[0], (list, tuple)) and len(sp_list[0]) >= 1:
                            # 兼容 (pos,...) 结构
                            sp_np = np.stack([to_vec3(sp[0]) for sp in sp_list], 0).astype(np.float32)
                        else:
                            sp_np = np.stack([to_vec3(sp) for sp in sp_list], 0).astype(np.float32)
                        g.create_dataset("shortest_path", data=sp_np, compression="gzip")

       
        # 3) 新格式：follow_groups（多候选，一份 JSON 最通用）
        if "follow_groups" in output:
            clean_groups = _pyify(output["follow_groups"])                     # <<< 关键：先净化
            groups_json = json.dumps(clean_groups, ensure_ascii=False).encode("utf-8")
            f.create_dataset("follow_groups_json", data=groups_json)
        else:
            f.create_dataset("follow_groups_json", data=json.dumps([],
                                ensure_ascii=False).encode("utf-8"))





        # 4) 可选：记录一些元信息
        for k in ["sample_fps", "plan_fps", "follow_size", "follow_result"]:
            if k in output:
                f.attrs[k] = output[k]

    print(f"✅  HDF5 saved to: {h5_path}")
