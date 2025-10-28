#!/usr/bin/env python3
"""
用法示例：
python3 rescale_stl.py --in vessel.stl --out vessel_small.stl --scale 0.5
"""
import argparse
from stl import mesh        # numpy-stl 提供的类
import numpy as np

def rescale_stl(in_path: str, out_path: str, scale: float):
    # 读取 STL
    m = mesh.Mesh.from_file(in_path)

    # 所有顶点坐标乘同一个系数即可等比例缩放
    m.vectors *= scale      # shape = (n_facets, 3, 3)

    # 保存
    m.save(out_path)
    print(f"Done. Saved scaled model (× {scale}) to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="等比例缩小 STL 模型")
    parser.add_argument("--in",  dest="in_path",  required=True, help="输入 .stl")
    parser.add_argument("--out", dest="out_path", required=True, help="输出 .stl")
    parser.add_argument("--scale", type=float, required=True,
                        help="缩放系数，0~1 表示缩小")
    args = parser.parse_args()
    rescale_stl(args.in_path, args.out_path, args.scale)
