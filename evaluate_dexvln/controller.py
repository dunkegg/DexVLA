import numpy as np

class ArcLengthPath:
    def __init__(self, xy_yaw_points):
        self.points = np.array(xy_yaw_points)  # shape = (N, 3)
        self.N = len(self.points)

        # 1. 计算每段的距离
        diffs = np.diff(self.points[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.s = np.zeros(self.N)
        self.s[1:] = np.cumsum(seg_lengths)  # 弧长累计
        self.s_total = self.s[-1]

    def _interp(self, s_query):
        """给定弧长 s_query，返回插值后的 (x, y, yaw)"""
        # 限制范围
        s_query = np.clip(s_query, 0, self.s_total)

        # 找到所在区间
        idx = np.searchsorted(self.s, s_query) - 1
        idx = np.clip(idx, 0, self.N - 2)

        # 插值比例
        t = (s_query - self.s[idx]) / (self.s[idx+1] - self.s[idx] + 1e-8)

        # x, y 线性插值
        x = (1-t)*self.points[idx, 0] + t*self.points[idx+1, 0]
        y = (1-t)*self.points[idx, 1] + t*self.points[idx+1, 1]

        # yaw 用插值防止跳变
        yaw0 = self.points[idx, 2]
        yaw1 = self.points[idx+1, 2]
        dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
        yaw = yaw0 + t * dyaw

        return x, y, yaw


class TrajectoryFollower:
    def __init__(self, path_xyyaw, total_time,
                 kp_xy=1.0, ki_xy=0.0, kd_xy=0.0,
                 kp_yaw=0.5, ki_yaw=0.0, kd_yaw=0.0):

        # 如果只有一个点 → 复制一下避免插值出错
        if len(path_xyyaw) == 1:
            path_xyyaw = [path_xyyaw[0], path_xyyaw[0]]

        self.path = ArcLengthPath(path_xyyaw)
        self.total_time = total_time
        self.start_time = None

        # xy PID 参数
        self.kp_xy = kp_xy
        self.ki_xy = ki_xy
        self.kd_xy = kd_xy

        # yaw PID 参数
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw

        # 状态变量
        self.integral_err_xy = np.zeros(2)
        self.prev_err_xy = np.zeros(2)

        self.integral_err_yaw = 0.0
        self.prev_err_yaw = 0.0

    def reset(self, start_time):
        self.start_time = start_time
        self.integral_err_xy[:] = 0
        self.prev_err_xy[:] = 0
        self.integral_err_yaw = 0.0
        self.prev_err_yaw = 0.0

    def step(self, current_time, x_cur, y_cur, yaw_cur):
        if self.start_time is None:
            self.start_time = current_time

        # 1. 计算当前参考位置（轨迹插值）
        elapsed = current_time - self.start_time
        progress = min(elapsed / self.total_time, 1.0)
        s_target = progress * self.path.s_total
        x_ref, y_ref, yaw_ref = self.path._interp(s_target)

        # ---------------- XY PID ----------------
        dx = x_ref - x_cur
        dy = y_ref - y_cur
        err_xy = np.array([dx, dy])

        self.integral_err_xy += err_xy
        derr_xy = err_xy - self.prev_err_xy
        self.prev_err_xy = err_xy

        correction_xy = (
            self.kp_xy * err_xy +
            self.ki_xy * self.integral_err_xy +
            self.kd_xy * derr_xy
        )

        # ---------------- Yaw PID ----------------
        dyaw = np.arctan2(np.sin(yaw_ref - yaw_cur), np.cos(yaw_ref - yaw_cur))

        self.integral_err_yaw += dyaw
        derr_yaw = dyaw - self.prev_err_yaw
        self.prev_err_yaw = dyaw

        correction_yaw = (
            self.kp_yaw * dyaw +
            self.ki_yaw * self.integral_err_yaw +
            self.kd_yaw * derr_yaw
        )

        # ---------------- 合成控制 ----------------
        x_cmd = x_ref + correction_xy[0]
        y_cmd = y_ref + correction_xy[1]

        # yaw 通常不能瞬间跳太快，加个限速
        max_yaw_delta = 0.05  # 每步最大转角 (rad)
        yaw_cmd = yaw_cur + np.clip(correction_yaw, -max_yaw_delta, max_yaw_delta)

        yaw_cmd = yaw_ref

        return x_cmd, y_cmd, yaw_cmd
