import subprocess
import time

cmd = "./scripts/train_dexvla_stage2_follow.sh"
cmd = "python eval_follow_whabitat_easy.py --yaml_file_path habitat_for_sim/cfg/exp_eval_multi.yaml"

while True:
    print(f"\n🚀 启动脚本：{cmd}")
    process = subprocess.Popen(cmd, shell=True)
    retcode = process.wait()  # 等脚本运行结束（无论成功或失败）

    if retcode == 0:
        print("✅ 脚本正常退出，任务完成。")
        break
    else:
        print(f"❌ 脚本异常退出（code={retcode}），5 秒后重新启动...")
        time.sleep(5)
