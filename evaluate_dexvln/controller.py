import numpy as np
import magnum as mn
import quaternion as qt
import time

class TrajectoryFollower:
    """
    path: [(mn.Vector3, mn.Quaternion, yaw), ...]  # global 3D + orientation
    local_path: [(x, y, yaw), ...]  # optional 2D/yaw local plan
    total_time: 轨迹总时长（秒）  —— 仅用于 global path 时间推进
    """
    def __init__(self, hb_path, local_path=None, total_time=5.0):
        if len(hb_path) < 2:
            raise ValueError("Global path must contain at least 2 points.")

        self.hb_path = hb_path
        self.local_path = local_path  # can be None or list of (x, y, yaw)
        self.total_time = float(total_time)
        self.start_time = None

        self._prepare_time_params()

    def _prepare_time_params(self):
        self.num_segments = len(self.hb_path) - 1
        self.seg_time = self.total_time / self.num_segments

    def reset(self, start_time):
        self.start_time = float(start_time)

    def step(self, current_time):
        """
        Returns:
            global_target: (mn.Vector3, mn.Quaternion, yaw)
            local_future_path: list of (x, y, yaw) from next local point onward, or None if no local_path
        """
        if self.start_time is None:
            raise RuntimeError("Call reset(start_time) before step().")

        elapsed = current_time - self.start_time

        # 1. global target
        if elapsed >= self.total_time:
            global_target = self.hb_path[-1]
        else:
            i = int(elapsed // self.seg_time)
            i = min(i, self.num_segments - 1)
            global_target = self.hb_path[i + 1]

        # 2. local future path
        local_future = None
        if self.local_path is not None and len(self.local_path) > 0:
            # 找到 local_path 中 “下一个点”的索引
            # 假设我们用相同 i 来对齐 local 和 global 步数 —— 若 local_path 长度 等于 global path 长度
            # 否则你也可以传入一个 local_index 来控制
            idx = i + 1
            if idx < len(self.local_path):
                local_future = self.local_path[idx : ]
            else:
                local_future = []  # 已经到 local_path 末尾

        return global_target, local_future
