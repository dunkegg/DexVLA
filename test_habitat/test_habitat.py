import habitat_sim
import magnum as mn
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "/wangzejin/goat_bench/data/scene_datasets/hm3d/train/00529-W9YAR9qcuvN/W9YAR9qcuvN.basis.glb"
sim_cfg.gpu_device_id = 0
sim_cfg.enable_physics = False

agent_cfg = habitat_sim.AgentConfiguration()

agent_cfg.sensor_specifications = []
sensor = habitat_sim.CameraSensorSpec()
sensor.sensor_type = habitat_sim.SensorType.COLOR
sensor.resolution = [480, 640]
sensor.uuid = "color"
sensor.position =  mn.Vector3(0.0, 1.5, 0.0)

agent_cfg.sensor_specifications = [sensor]

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
print("✅ Simulator init success")

# import habitat_sim
# import habitat_sim.utils.common as utils
# import numpy as np
# import imageio

# def create_simulator(scene_path):
#     # Simulator config
#     sim_cfg = habitat_sim.SimulatorConfiguration()
#     sim_cfg.scene_id = scene_path
#     sim_cfg.enable_physics = False
#     sim_cfg.gpu_device_id = 0  # 若有多块 GPU，可改成其他编号
#     agent_cfg = habitat_sim.AgentConfiguration()
#     # Simulator 初始化，不提供 agent
#     cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
#     sim = habitat_sim.Simulator(cfg)

#     # 创建传感器参数
#     sensor_spec = habitat_sim.CameraSensorSpec()
#     sensor_spec.uuid = "color"
#     sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
#     sensor_spec.resolution = [480, 640]
#     sensor_spec.position = [0.0, 1.5, 0.0]
#     sensor_spec.orientation = [0.0, 0.0, 0.0]
#     sensor_spec.hfov = 90

#     # 获取 root 节点并挂载 sensor
#     root_node = sim.get_active_scene_graph().get_root_node()
#     sensor_node = root_node.create_child()
#     sim._sensors[sensor_spec.uuid] = sim.create_sensor(sensor_spec, sensor_node)

#     return sim, sensor_node

# # 替换成你的场景文件路径
# scene_path = "/wangzejin/goat_bench/data/scene_datasets/hm3d/train/00529-W9YAR9qcuvN/W9YAR9qcuvN.basis.glb"
# sim, sensor_node = create_simulator(scene_path)

# # 渲染并保存图像
# obs = sim._sensors["color"].get_observation()
# imageio.imwrite("output_rgb.png", obs)
# print("✅ 成功渲染图像并保存至 output_rgb.png")

# sim.close()
