import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘制2D图形
import numpy as np  # 导入numpy库，用于进行数值计算和数组操作
import warnings  # 导入warnings模块，用于处理警告信息（如忽略或显示警告）
from scipy.spatial import transform  # 从scipy的spatial模块导入transform，用于处理空间旋转（如四元数转矩阵）


def quat_to_mat(quat):
    "assumes quat is in wxyz format"  # 假设输入的四元数是wxyz格式（w为实部，x、y、z为虚部）
    # 如果四元数是单位四元数（表示无旋转），直接返回3x3单位矩阵
    if quat[0] == 1 and quat[1] == 0 and quat[2] == 0 and quat[3] == 0:
        return np.eye(3)
    else:
        # 将wxyz格式转换为scipy默认的xyzw格式（因为scipy的from_quat函数要求输入为xyzw）
        quat = [quat[3], quat[0], quat[1], quat[2]]
        # 将四元数转换为旋转矩阵，并转置（调整坐标系对齐）
        return transform.Rotation.from_quat(quat).as_matrix().T


def create_camera_matrix(
        image_size: int,
        pos: list,  # 相机外参：三维位置（x,y,z）
        quat: list,  # 相机外参：三维姿态（四元数wxyz）
        fov: float = 45.0,  # 相机内参：视场角（决定视野范围宽窄）
        debug=False
) -> np.ndarray:
    pos = np.array(pos)  # 相机位置转为数组

    # 1. 平移矩阵（4x4）：将世界坐标系原点平移到相机坐标系（抵消相机位置）
    # 原理：三维空间中，相机看到的点 = 世界坐标 - 相机位置（即世界坐标 + (-相机位置)）
    translation = np.eye(4)  # 初始化4x4单位矩阵
    translation[0:3, 3] = -pos  # 前3行第4列存储相机位置的负值（实现平移）

    # 2. 旋转矩阵（4x4）：将世界坐标系旋转到相机坐标系（匹配相机朝向）
    # 原理：相机姿态通过四元数描述，需转换为旋转矩阵，使世界坐标与相机视角对齐
    R = quat_to_mat(quat)  # 调用quat_to_mat将四元数转为3x3旋转矩阵
    rotation = np.eye(4)  # 初始化4x4单位矩阵
    rotation[0:3, 0:3] = R  # 前3x3部分替换为旋转矩阵（处理姿态）

    # 3. 焦距矩阵（3x4）：将三维相机坐标投影到二维图像平面（缩放作用）
    # 原理：视场角越大，视野越广，相同三维距离在图像上的像素距离越短（缩放因子越小）
    focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * (image_size / 2.0)  # 计算缩放因子
    # 构建3x4焦距矩阵（x方向取负号：纠正图像x轴与世界坐标x轴的方向差异）
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # 4. 图像中心矩阵（3x3）：将投影后的坐标偏移到图像中心（匹配像素坐标系）
    # 原理：像素坐标系原点在图像左上角，需偏移到中心（(image_size-1)/2, (image_size-1)/2）
    image = np.eye(3)  # 初始化3x3单位矩阵
    image[0, 2] = (image_size - 1) / 2.0  # x方向偏移（左→右中心）
    image[1, 2] = (image_size - 1) / 2.0  # y方向偏移（上→下中心）

    # 组合矩阵：相机矩阵 = 图像中心矩阵 × 焦距矩阵 × 旋转矩阵 × 平移矩阵
    # 计算顺序：从右到左（先平移→旋转→缩放→偏移）
    camera_matrix = image @ focal @ rotation @ translation

    if debug:  # 调试模式下打印各矩阵，验证计算是否正确
        print("平移矩阵（抵消相机位置）：\n", translation[:3, 3])
        print("旋转矩阵（匹配相机姿态）：\n", rotation[:3, :3])
        print("焦距矩阵（缩放投影）：\n", focal)
        print("图像中心矩阵（偏移到中心）：\n", image)
    return camera_matrix  # 输出3x4相机矩阵，用于后续三维→二维转换


def point2pixel(
    points: np.ndarray,
    camera_matrix: np.ndarray = None,
    camera_kwargs: dict = dict(image_size=80),
) -> np.ndarray:
    """Transforms from world coordinates to pixel coordinates.

    Args:
      point: np.ndarray: the point to be transformed.
      camera_matrix: np.ndarray: the camera matrix. If None, it is generated from the camera_kwargs.
      camera_kwargs: dict:  (Default value = dict(image_size=80))

    Returns:
        np.ndarray : the pixel coordinates of the point.
    """
    # 如果未提供相机矩阵，使用camera_kwargs参数生成一个
    if camera_matrix is None:
        camera_matrix = create_camera_matrix(** camera_kwargs)

    # 如果输入的点是1维数组（单个点），增加一个维度变为2D数组（Nx3格式）
    if points.ndim == 1:
        points = points[np.newaxis, :]

    # 将3D点转换为齐次坐标（Nx3 -> Nx4）：增加一列全为1的维度，方便矩阵乘法
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # 投影到像素坐标：用相机矩阵乘以齐次坐标（3x4矩阵 × 4xN矩阵 = 3xN矩阵）
    pixel_coords = camera_matrix.dot(points_homogeneous.T)

    # 齐次坐标转笛卡尔坐标：除以最后一个分量（归一化）
    pixel_coords = pixel_coords / pixel_coords[-1, :]
    # 取前两维（x、y像素坐标），四舍五入并转换为整数，返回结果
    pixel_coords = np.round(pixel_coords[:-1, :].T).astype(np.int32)

    return pixel_coords.squeeze()  # 移除多余的维度（如单个点时返回1D数组）


def plot_3D_to_2D(
    ax: plt.Axes,
    data: np.ndarray,
    base_image: np.ndarray = None,
    add_line: bool = True,
    image_size: int = 80,
    line_kwargs: dict = dict(color="black"),
    scatter_kwargs: dict = dict(color="blue"),
) -> plt.Axes:
    """Plot 3D data to 2D image

    Args:
        ax (plt.Axes): The axes to plot on
        data (np.ndarray): The data to plot
        base_image (np.ndarray): An image to plot behind the data
        add_line (bool): If True, add a line between the points
        image_size (int): Size of the image. If base_image is provided, this is overwritten
        line_kwargs (dict): Keyword arguments for the line. See plt.plot
        scatter_kwargs (dict): Keyword arguments for the scatter. See plt.scatter
    """
    # 如果提供了背景图像，将图像上下翻转（因为图像坐标系y轴方向与默认相反）并显示
    if base_image is not None:
        base_image = np.flipud(base_image)
        plt.imshow(base_image)
        image_size = base_image.shape[0]  # 用背景图像的尺寸覆盖输入的image_size
        warnings.warn("Image size overwritten by base_image")  # 发出警告：图像尺寸已被覆盖

    # 设置坐标轴范围为图像尺寸（0到image_size）
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    # 如果数据是3D的（每个点有x、y、z三个坐标）
    if len(data[0]) == 3:
        # 将每个3D点转换为2D像素坐标（沿第1轴应用point2pixel函数）
        data = np.apply_along_axis(
            point2pixel, 1, data, camera_kwargs=dict(image_size=image_size)
        )
        # 过滤掉超出图像范围的点（只保留x、y都在0到image_size之间的点）
        data = [point for point in data if np.all((0 <= point) & (point <= image_size))]
        data = np.array(data)  # 转换回numpy数组
        data[:, 1] = image_size - data[:, 1]  # 调整y坐标（纠正图像坐标系y轴方向）
    # 在2D坐标轴上绘制散点图
    ax.scatter(data[:, 0], data[:, 1], **scatter_kwargs)
    # 如果add_line为True，在点之间绘制连线
    if add_line:
        ax.plot(data[:, 0], data[:, 1], **line_kwargs)


def plot_w_mesh(mesh, points: np.ndarray, **kwargs):
    """
    Plot a mesh with points

    Args:
        mesh: The mesh to plot
        points: The points to plot
        **kwargs: The keyword arguments to pass to the scatter function

    """
    fig = plt.figure()  # 创建一个图形对象
    ax = fig.add_subplot(111, projection="3d")  # 添加一个3D子图
    # 设置坐标轴范围为网格的边界（x、y、z轴分别对应网格的最小和最大值）
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    # 在3D空间中绘制点（这里原代码可能有误，x、y、z坐标都用了points[:,0]，应为points[:,0], points[:,1], points[:,2]）
    ax.scatter(points[:, 0], points[:, 0], points[:, 0],** kwargs)
    plt.show()  # 显示图形
