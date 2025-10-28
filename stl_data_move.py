import trimesh
import numpy as np

# 1) 读取
mesh = trimesh.load('/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery_0912_200k.stl')

# 2) 平移向量：+X, +Y, -Z
dx, dy, dz = -93.0, 44.0, 0        # 按需修改数值
mesh.apply_translation([dx, dy, dz])

# 3) 保存
mesh.export('/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery_0912_200k_move.stl')