import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.spatial import transform


def quat_to_mat(quat):
    "assumes quat is in wxyz format"
    if quat[0] == 1 and quat[1] == 0 and quat[2] == 0 and quat[3] == 0:
        return np.eye(3)
    else:
        quat = [quat[3], quat[0], quat[1], quat[2]]
        return transform.Rotation.from_quat(quat).as_matrix().T


def create_camera_matrix(
    image_size: int,
    pos: list,
    quat: list,
    fov: float = 45.0,
    debug=False
) -> np.ndarray:
    pos = np.array(pos)

    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    R = quat_to_mat(quat)
    rotation = np.eye(4)
    rotation[0:3, 0:3] = R

    # Focal transformation matrix (3x4).
    focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * (image_size / 2.0)
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    # Focal transformation matrix (3x4).

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (image_size - 1) / 2.0
    image[1, 2] = (image_size - 1) / 2.0

    # Compute the 3x4 camera matrix.
    camera_matrix = image @ focal @ rotation @ translation
    if debug:
        print("quat: \n", quat)
        print("translation: \n", translation[:3, 3])
        print("rotation: \n", rotation[:3, :3])
        print("focal: \n", focal)
        print("image: \n", image)
    return camera_matrix


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
    if camera_matrix is None:
        camera_matrix = create_camera_matrix(**camera_kwargs)

    if points.ndim == 1:
        points = points[np.newaxis, :]

    # Making points homogeneous
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Projecting to pixel coordinates
    pixel_coords = camera_matrix.dot(points_homogeneous.T)

    # Converting homogeneous coordinates to Cartesian coordinates
    pixel_coords = pixel_coords / pixel_coords[-1, :]
    pixel_coords = np.round(pixel_coords[:-1, :].T).astype(np.int32)

    return pixel_coords.squeeze()


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
    if base_image is not None:
        base_image = np.flipud(base_image)
        plt.imshow(base_image)
        image_size = base_image.shape[0]
        warnings.warn("Image size overwritten by base_image")

    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    if len(data[0]) == 3:
        data = np.apply_along_axis(
            point2pixel, 1, data, camera_kwargs=dict(image_size=image_size)
        )
        data = [point for point in data if np.all((0 <= point) & (point <= image_size))]
        data = np.array(data)  # Convert back to numpy array
        data[:, 1] = image_size - data[:, 1]
    ax.scatter(data[:, 0], data[:, 1], **scatter_kwargs)
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    ax.scatter(points[:, 0], points[:, 0], points[:, 0], **kwargs)
    plt.show()
