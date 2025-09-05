import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import magnum as mn
import math
import random
from omegaconf import DictConfig
from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)

def habitat_quat_to_magnum(q_np):
    # q_np: numpy.quaternion (w, x, y, z)
    return mn.Quaternion(
        mn.Vector3(q_np.x, q_np.y, q_np.z),
        q_np.w
    )

def wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def local2world(rel_local: np.ndarray,
                follow_pos: np.ndarray,
                follow_quat: mn.Quaternion,
                follow_yaw: float,
                type: int) -> np.ndarray:
    """
    rel_local: (N, 4), 每行是 [x, y, z, yaw_local]
    follow_pos: (3,), 当前机器人在世界坐标系的位置
    follow_quat: mn.Quaternion，当前机器人姿态（旋转四元数）
    follow_yaw: float, 当前机器人朝向角（rad）
    type: int，对称类型（用于恢复左右翻转）

    return: (N, 4)，世界坐标路径 [x_world, y_world, z_world, yaw_world]
    """

    out = rel_local.copy()
    R = follow_quat  # 用于局部 → 世界的旋转
    for i, row in enumerate(rel_local):
        v_local = mn.Vector3(*row[:3])
        if type == 1:
            # 和之前对称操作相反
            v_local = mn.Vector3(v_local.x, v_local.y, -v_local.z)
        else:
            v_local = mn.Vector3(-v_local.x, v_local.y, -v_local.z)

        # 旋转 + 平移
        v_world = R.transform_vector(v_local) + mn.Vector3(*follow_pos)
        out[i, :3] = [v_world.x, v_world.y, v_world.z]

        # yaw 反变换
        yaw_local = row[3]
        # yaw_world = wrap_pi(yaw_local + follow_yaw)
        yaw_world = yaw_local + follow_yaw
        out[i, 3] = yaw_world

    return out.astype(np.float32)

def load_humanoid(sim):
    names = ["female_0", "female_1", "female_2", "female_3", "male_0", "male_1", "male_2", "male_3"]
    humanoid_name =  random.choice(names) 
    data_root = "/home/wangzejin/habitat/ON-MLLM/human_follower/habitat_humanoids" #wzjpath
    urdf_path = f"{data_root}/{humanoid_name}/{humanoid_name}.urdf"
    motion_pkl = f"{data_root}/{humanoid_name}/{humanoid_name}_motion_data_smplx.pkl"

    agent_cfg = DictConfig(
        {
            "articulated_agent_urdf": urdf_path,
            "motion_data_path": motion_pkl,
            "auto_update_sensor_transform": True,
        }
    )
    humanoid = KinematicHumanoid(agent_cfg, sim)
    humanoid.reconfigure()
    humanoid.update()

    controller = HumanoidRearrangeController(walk_pose_path=motion_pkl)
    controller.reset(humanoid.base_transformation)

    return humanoid, controller

def shortest_angle_diff(a, b):
    """
    返回 b−a 的最短角差，范围 (-π, π]
    """
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return diff



def to_vec3(v) -> mn.Vector3:
    """接受 magnum.Vector3 或 list/tuple/np.ndarray"""
    if isinstance(v, mn.Vector3):
        return v
    return mn.Vector3(float(v[0]), float(v[1]), float(v[2]))

# # ---------- 辅助转换 ---------- #
# def to_vec3(arr_like):
#     """
#     任意 Vector3 表示 → np.float32[3]
#     支持：Magnum Vector3 / ndarray / list / tuple
#     """
#     if isinstance(arr_like, mn.Vector3):
#         return np.array([arr_like.x, arr_like.y, arr_like.z], dtype=np.float32)
#     arr = np.asarray(arr_like, dtype=np.float32).reshape(3)
#     return arr

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