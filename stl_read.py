import trimesh

# mesh = trimesh.load('/home/xingenming/Downloads/cathsim/src/cathsim/dm/components/phantom_assets/meshes/artery0002_aligned_rot180_move_origin/hull_0.stl')
mesh = trimesh.load('/home/xingenming/Downloads/cathsim/src/cathsim/dm/components/phantom_assets/meshes/phantom3/hull_0.stl')

print('三角形数量:', len(mesh.faces))
print('=== 三角形信息 ===')
for idx, (v0, v1, v2) in enumerate(mesh.triangles):
    print(f'三角形 #{idx}')
    print(f'  法向量 : {mesh.face_normals[idx]}')
    print(f'  顶点1  : {v0}')
    print(f'  顶点2  : {v1}')
    print(f'  顶点3  : {v2}')
    print('-' * 30)