import trimesh
import numpy as np

# 1) 读取
mesh = trimesh.load('/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery_0912_200k_move.stl')

# 2) 绕 Y 正半轴旋转 30°（可改成任何角度）
angle = np.deg2rad(270)          # 30° 示例
axis  = [1, 0, 0]               # Y 正半轴
R = trimesh.transformations.rotation_matrix(angle, axis)

# 3) 以世界原点为中心旋转
mesh.apply_transform(R)

# 4) 保存
mesh.export('/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery_0912_200k_move_rotate_x270.stl')