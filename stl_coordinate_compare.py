# import trimesh

# def show_frame(stl_path, name):
#     mesh = trimesh.load(stl_path)
#     bounds = mesh.bounds          # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
#     print(f"{name}  AABB：")
#     print(f"  min = {bounds[0]}")
#     print(f"  max = {bounds[1]}")
#     print(f"  center = {(bounds[0] + bounds[1]) / 2}")
#     print()

# show_frame("/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/phantom3.stl", "A")
# show_frame("/home/xingenming/Downloads/artery.stl", "B") # 将原始分割数据直接导出stl
# show_frame("/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery20250829.stl", "C") #将原始分割数据裁剪掉右上角的血管，然后导出stl
# show_frame("/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery20250829_rotate_x270.stl", "D")#将原始分割数据裁剪掉右上角的血管，然后导出stl，然后沿着x轴旋转270度
# # A  AABB：
# #   min = [-0.07343519 -0.02139059 -0.00983204]
# #   max = [0.01792319 0.1859289  0.07134185]
# #   center = [-0.027756    0.08226915  0.03075491]

# # B  AABB：
# #   min = [  48.93759155 -138.85163879   -0.23415115]
# #   max = [135.57351685   0.23415115 179.59393311]
# #   center = [ 92.2555542  -69.30874382  89.67989098]

# # C  AABB：
# #   min = [  56.89872742 -130.42219543   -0.23415115]
# #   max = [109.8168869  -40.50814819 101.85575104]
# #   center = [ 83.35780716 -85.46517181  50.81079994]

# # D  AABB：
# #   min = [56.89872742 -0.23415115 40.50814819]
# #   max = [109.8168869  101.85575104 130.42219543]
# #   center = [83.35780716 50.81079994 85.46517181]


import trimesh
import numpy as np

# 1) 读取
A = trimesh.load("/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/phantom3.stl")
B = trimesh.load("/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery20250829.stl")

# 2) 整体对齐（ICP：平移 + 旋转 + 可选缩放）
#    scale=False -> 只做刚体变换；True -> 允许整体缩放
matrix, cost = B.register(A, scale=False)   # 4×4 齐次矩阵
B_aligned = B.copy().apply_transform(matrix)

# 2) 再绕 X 轴旋转 180
angle_x180 = np.deg2rad(180)          
x_180 = trimesh.transformations.rotation_matrix(angle_x180, [1, 0, 0])
B_aligned_rot180 = B_aligned.copy().apply_transform(x_180)

# 3) 保存对齐后的文件
B_aligned_rot180.export("/home/xingenming/Downloads/cathsim/stl2mjcf-main/stl2mjcf/artery20250829_aligned_rot180.stl")

# 4) 验证
print("A 中心:", A.centroid)
print("B 对齐后中心:", B_aligned_rot180.centroid)
