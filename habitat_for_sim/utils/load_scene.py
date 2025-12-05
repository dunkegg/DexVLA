import habitat_sim
from habitat_for_sim.utils.goat import find_scene_path, calculate_euclidean_distance
from habitat_for_sim.utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_habitat_to_normal,
)
from habitat_for_sim.agent.path_generator import generate_path
from human_follower.walk_behavior import get_path_with_time
import magnum as mn
import numpy as np

def load_simulator(cfg):
    scene_mesh_dir, _ = find_scene_path(cfg, cfg.current_scene)
    sim_settings = {
        "scene": scene_mesh_dir,
        "default_agent": [0],
        "sensor_height": 1.5,
        "width": cfg.img_width,
        "height": cfg.img_height,
        "hfov": cfg.hfov,
    }
    sim_cfg = make_simple_cfg(sim_settings)
    simulator = habitat_sim.Simulator(sim_cfg)
    
    # 从 sim_cfg 中获取 agent 配置
    agent_cfg = sim_cfg.agents[0]  # 获取默认代理的配置
    # 获取 NavMeshSettings 对象
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.agent_radius = agent_cfg.radius             # 设置 agent 碰撞半径
    navmesh_settings.agent_height = agent_cfg.height             # 设置 agent 碰撞高度
    navmesh_settings.agent_max_climb = 1          # 设置最大爬升高度
    navmesh_settings.agent_max_slope = 45.0         # 设置最大坡度角度（单位：度）

    # 重新生成导航网格
    navmesh_success = simulator.recompute_navmesh(simulator.pathfinder, navmesh_settings)

    # 验证导航网格是否成功生成
    if not navmesh_success or not simulator.pathfinder.is_loaded:
        raise RuntimeError("Navmesh recomputation failed. Cannot proceed with pathfinding.")
    
    return simulator

def generate_path_from_scene(obj_data, pathfinder,min_distance = 5 ,human_fps = 5, human_speed = 0.7):
    # 设置起始位置和旋转
    start_position = obj_data.start_position
    start_rotation = obj_data.start_rotation
    distance = obj_data.info['euclidean_distance']
    goal_position = obj_data.goal["position"]

    start_normal = pos_habitat_to_normal(start_position)
    start_floor_height = start_normal[-1]



    if goal_position is None or start_position is None:
        return None
    # 使用 ShortestPath 对象生成避免穿墙的最短路径
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_position
    shortest_path.requested_end = goal_position

    # 查找最短路径 # 模拟前沿探索过程
    if pathfinder.find_path(shortest_path):
        
        # 检查起始点和目标点是否在同一楼层 (高度差小于1m)
        start_floor_height = start_position[1]  # y 值代表高度
        goal_floor_height = goal_position[1]
        if abs(start_floor_height - goal_floor_height) > 1:
            print(f"Skipping episode due to height difference: {start_floor_height} vs {goal_floor_height}")
            return None
        path = shortest_path.points
        # all_distance = 0
        # for i in range(len(path)-1):
        #     distance = calculate_euclidean_distance(path[i], path[i+1])
        #     all_distance+=distance
        
        # if all_distance < min_distance:
        #     # print(f"Skipping episode due to short distance: {all_distance}m")
        #     return None
        

        # 检查路径是否跨楼层 (高度差小于1m)
        floor_heights = [point[1] for point in path]  # 获取所有路径点的高度
        if max(floor_heights) - min(floor_heights) > 1:
            print("Skipping episode due to multi-floor path")
            return None
        
        # # 初始化探索类
        # if not cfg.shortest_path:

        #     num_frontiers = random.randint(1, 4)
        #     #print("original path:",path)
        #     explorer = FrontierExploration(simulator)
        #     explorer.explore_until_target(
        #                 start_position = start_position,
        #                 target_position = goal_position,
        #                 num_frontiers = num_frontiers)
        #     path = explorer.trail
        #print("frontier_exploration path:", path)
        

        #轨迹优化流水线
        new_path = generate_path(path, pathfinder,filt_distance=0.1, num_points_between = 5, resolution=1024,visualize=False)
        
        #print(path[0])
        #print(path)
    else:
        # print("No valid path found:", obj_data["object_category"])
        return None 
    

    

    dense_path = get_path_with_time(new_path, time_step=1/human_fps, speed=human_speed)
    
    # 修正高度偏移
    height_bias = 0
    for i in range(len(dense_path)):
        pos, quat, yaw = dense_path[i]
        new_pos = mn.Vector3(pos.x, pos.y - height_bias, pos.z)
        dense_path[i] = (new_pos,quat,yaw)

    # 插入高度跳变检测（连续跳变次数超过阈值才判为异常）
    height_jump_threshold = 0.04
    max_allowed_jumps = 5
    jump_count = 0
    normal_height = 0.083759
    for i in range(1, len(dense_path)):
        curr_height = dense_path[i][0].y

        if curr_height - normal_height > height_jump_threshold:
            jump_count += 1
            if jump_count >= max_allowed_jumps:
                break

    if jump_count >= max_allowed_jumps:
        print(f"Skipping episode due to {jump_count} large height jumps.")
        return None
    
    return dense_path

def generate_segment(start_position, goal_position, pathfinder, human_fps=5, human_speed=0.7, check_height = True):
    # Shortest path
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_position
    shortest_path.requested_end = goal_position


    if not pathfinder.find_path(shortest_path):
        if check_height:
            return None
        else:
            shortest_path.points = [start_position, goal_position]

    direction = shortest_path.points[-1] - shortest_path.points[0] 
    if np.linalg.norm(direction) < 1e-4:
        shortest_path.points = [start_position, goal_position]
    # Floor consistency check
    start_floor = start_position[1]
    goal_floor = goal_position[1]
    if abs(start_floor - goal_floor) > 1 and check_height:
        return None

    floor_heights = [p[1] for p in shortest_path.points]
    if max(floor_heights) - min(floor_heights) > 1 and check_height: 
        return None

    # Expand/optimize path
    new_path = generate_path(
        shortest_path.points, 
        pathfinder,
        num_points_between=1,
        resolution=256,
        filt_distance=1e-4,
        visualize=False,
        opt=False
    )

    dense_path = get_path_with_time(
        new_path,
        time_step=1/human_fps,
        speed=human_speed
    )

    # Height smoothing (same as your logic)
    height_bias = 0
    for i in range(len(dense_path)):
        pos, quat, yaw = dense_path[i]
        new_pos = mn.Vector3(pos.x, pos.y - height_bias, pos.z)
        dense_path[i] = (new_pos, quat, yaw)

    # Height jump detection
    height_jump_threshold = 0.04
    max_allowed_jumps = 5
    jump_count = 0
    normal_height = 0.083759

    for i in range(1, len(dense_path)):
        curr_height = dense_path[i][0].y
        if curr_height - normal_height > height_jump_threshold:
            jump_count += 1
            if jump_count >= max_allowed_jumps and check_height:
                return None

    return dense_path

def generate_full_path_from_coords_with_index(coords, pathfinder, human_fps=5, human_speed=0.7):
    full_path = []
    index_map = []  # index_map[i] = full_path indices corresponding to coords[i]→coords[i+1]

    current_start_idx = 0  # full_path 的起始下标

    for i in range(len(coords) - 1):
        start_pos = coords[i]
        end_pos = coords[i + 1]

        segment = generate_segment(
            start_pos,
            end_pos,
            pathfinder,
            human_fps,
            human_speed
        )

        if segment is None:
            print(f"Skipping segment {i}: no valid path from {start_pos} to {end_pos}")
            return None, None

        # Avoid duplicate first point
        if full_path:
            segment = segment[1:]

        # segment 的长度
        seg_len = len(segment)

        # full_path 拼接
        full_path.extend(segment)

        # 记录 segment 在 full_path 中的 start/end index ⭐
        seg_start = current_start_idx
        seg_end = current_start_idx + seg_len - 1

        index_map.append((seg_start, seg_end))

        current_start_idx += seg_len

    return full_path, index_map

def quat_to_yaw(q):
    x, y, z, w = q
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return np.arctan2(siny_cosp, cosy_cosp)


def shortest_angle_diff(a, b):
    """返回两个角的最小差值（范围：[-pi, pi]）"""
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff

def generate_path_from_scene_for_obj(obj_data, pathfinder, min_distance = 5 ,human_fps = 5, human_speed = 0.7):
    # 设置起始位置和旋转
    start_position = obj_data["start"]["position"]
    start_rotation = obj_data["goal"]["rotation"]
    goal_position = obj_data["goal"]["position"]
    goal_rotation = obj_data["goal"]["rotation"]

    start_normal = pos_habitat_to_normal(start_position)
    start_floor_height = start_normal[-1]

    if goal_position is None or start_position is None:
        return None
    # 使用 ShortestPath 对象生成避免穿墙的最短路径
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_position
    shortest_path.requested_end = goal_position

    # 查找最短路径 # 模拟前沿探索过程
    if pathfinder.find_path(shortest_path):
        
        # 检查起始点和目标点是否在同一楼层 (高度差小于1m)
        start_floor_height = start_position[1]  # y 值代表高度
        goal_floor_height = goal_position[1]
        if abs(start_floor_height - goal_floor_height) > 1:
            print(f"Skipping episode due to height difference: {start_floor_height} vs {goal_floor_height}")
            return None
        path = shortest_path.points

        # path = [start_position] + path + [goal_position]

        floor_heights = [point[1] for point in path]  # 获取所有路径点的高度
        if max(floor_heights) - min(floor_heights) > 1:
            print("Skipping episode due to multi-floor path")
            return None
        
        #轨迹优化流水线
        new_path = generate_path(path, pathfinder, filt_distance=0.1, num_points_between = 5, resolution=1024,visualize=False)

    else:
        return None 

    dense_path = get_path_with_time(new_path, time_step=1/human_fps, speed=human_speed)
    
    if dense_path:
        final_quat = dense_path[-1][1]
        final_yaw  = dense_path[-1][2]

        goal_yaw = quat_to_yaw(goal_rotation)

        yaw_diff = abs(shortest_angle_diff(final_yaw, goal_yaw))

        if yaw_diff > (np.pi / 2):    # 超过 90°
            print(f"Skipping episode due to final yaw difference: {yaw_diff * 180/np.pi:.2f}°")
            return None
    
    height_bias = 0
    for i in range(len(dense_path)):
        pos, quat, yaw = dense_path[i]
        new_pos = mn.Vector3(pos.x, pos.y - height_bias, pos.z)
        dense_path[i] = (new_pos,quat,yaw)
    
    # 插入高度跳变检测（连续跳变次数超过阈值才判为异常）
    height_jump_threshold = 0.04
    max_allowed_jumps = 5
    jump_count = 0
    normal_height = 0.083759
    # for i in range(1, len(dense_path)):
    #     curr_height = dense_path[i][0].y

    #     if curr_height - normal_height > height_jump_threshold:
    #         jump_count += 1
    #         if jump_count >= max_allowed_jumps:
    #             break

    # if jump_count >= max_allowed_jumps:
    #     print(f"Skipping episode due to {jump_count} large height jumps.")
    #     return None
    
    return dense_path
