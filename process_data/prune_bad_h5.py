#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import glob
import shutil
import h5py

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # 无 tqdm 时降级

DEFAULT_REQUIRED = [
    "action",
    "language_raw",
    "observations/history_images",
    "observations/images/cam_high",
    "observations/qpos",
]

def exists(h5f, path: str) -> bool:
    """检查 HDF5 内路径是否存在"""
    try:
        _ = h5f[path]
        return True
    except KeyError:
        return False

def check_file(h5_path: str, required_paths):
    """返回 (is_ok, missing_list, error_msg)"""
    try:
        with h5py.File(h5_path, "r") as f:
            missing = [p for p in required_paths if not exists(f, p)]
            return (len(missing) == 0, missing, None)
    except Exception as e:
        # 打不开文件也算坏样本
        return (False, ["<open_error>"], str(e))

def main():
    ap = argparse.ArgumentParser(
        description="Scan HDF5 and remove/move files missing required datasets."
    )
    ap.add_argument(
        "root",
        help="要扫描的目录（例如 data/split_data/multi_follow）或单个文件路径",
    )
    ap.add_argument(
        "--pattern", default="*.hdf5",
        help="匹配的文件通配符，默认 *.hdf5"
    )
    ap.add_argument(
        "--required",
        default=",".join(DEFAULT_REQUIRED),
        help="必需字段，逗号分隔（默认：%s）" % ",".join(DEFAULT_REQUIRED),
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="仅扫描并打印结果，不进行移动/删除"
    )
    ap.add_argument(
        "--delete", action="store_true",
        help="直接删除坏样本（危险操作，和 --move-to 互斥）"
    )
    ap.add_argument(
        "--move-to", default=None,
        help="将坏样本移动到该目录（默认创建 root/_bad_h5）。与 --delete 互斥"
    )
    ap.add_argument(
        "--good-list", default=None,
        help="把通过检查的文件列表写到此路径（如 good_files.txt）"
    )
    ap.add_argument(
        "--bad-list", default=None,
        help="把不通过的文件列表写到此路径（如 bad_files.txt）"
    )

    args = ap.parse_args()

    # 解析 required 列表
    required_paths = [s.strip() for s in args.required.split(",") if s.strip()]

    # 组装文件列表
    if os.path.isfile(args.root):
        files = [args.root]
        root_dir = os.path.dirname(os.path.abspath(args.root)) or "."
    else:
        root_dir = os.path.abspath(args.root)
        files = sorted(glob.glob(os.path.join(root_dir, args.pattern)))

    if not files:
        print(f"[WARN] 没找到文件：{args.root} (pattern={args.pattern})")
        return 0

    # 检查互斥参数
    if args.delete and args.move_to:
        print("[ERROR] --delete 与 --move-to 不能同时使用")
        return 2

    # 默认隔离目录
    move_dir = args.move_to
    if (not args.delete) and (move_dir is None) and (not args.dry_run):
        move_dir = os.path.join(root_dir, "_bad_h5")

    if move_dir:
        os.makedirs(move_dir, exist_ok=True)

    good, bad = [], []

    print(f"[INFO] 扫描目录：{root_dir}")
    print(f"[INFO] 检查字段：{required_paths}")
    print(f"[INFO] 文件数：{len(files)}")
    print(f"[INFO] 模式：{'干跑(dry-run)' if args.dry_run else ('删除坏样本' if args.delete else ('移动坏样本到 '+move_dir))}")
    print("-" * 60)

    for fp in tqdm(files):
        ok, missing, err = check_file(fp, required_paths)
        if ok:
            good.append(fp)
        else:
            reason = f"缺失: {missing}" if err is None else f"错误: {err}; 缺失: {missing}"
            print(f"[BAD] {os.path.basename(fp)} -> {reason}")
            bad.append(fp)
            if not args.dry_run:
                if args.delete:
                    try:
                        os.remove(fp)
                    except Exception as e:
                        print(f"[ERR] 删除失败 {fp}: {e}")
                else:
                    try:
                        shutil.move(fp, os.path.join(move_dir, os.path.basename(fp)))
                    except Exception as e:
                        print(f"[ERR] 移动失败 {fp}: {e}")

    print("-" * 60)
    print(f"[SUMMARY] 通过: {len(good)}  |  不通过: {len(bad)}")

    if args.good_list:
        with open(args.good_list, "w", encoding="utf-8") as f:
            for x in good:
                f.write(x + "\n")
        print(f"[INFO] 已写入 good 列表：{args.good_list}")

    if args.bad_list:
        with open(args.bad_list, "w", encoding="utf-8") as f:
            for x in bad:
                f.write(x + "\n")
        print(f"[INFO] 已写入 bad 列表：{args.bad_list}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
