export HABITAT_SIM_EGL=1
export MAGNUM_LOG=EGL
export HABITAT_SIM_GPU_DEVICE_ID=0     # 如果你用的是第 0 个显卡
export CUDA_VISIBLE_DEVICES=0
export MAGNUM_LOG=verbose MAGNUM_GPU_VALIDATION=ON
python test_habitat.py