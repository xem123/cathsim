import trimesh
import numpy as np

mesh = trimesh.load('/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery20250829_rotate_x270.stl')

# R = trimesh.transformations.rotation_matrix(np.pi/2 + np.pi, [1, 0, 0]) #  绕 x 轴旋转 270°
angle = np.deg2rad(30)
R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0]) #  绕 x 轴旋转 270°

mesh.apply_transform(R)

mesh.export('/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery20250829_rotate_x270_y30.stl')