"""Wrappers for dm_control environments to be used with OpenAI gym."""

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from cathsim.dm import make_dm_env
from dm_env import specs
from gymnasium.envs.registration import EnvSpec
import os
from PIL import Image
import cv2
from pathlib import Path
from cv2 import MORPH_ELLIPSE, getStructuringElement, morphologyEx
from cathsim.dm.visualization import (
    create_camera_matrix,
    point2pixel,
)

def convert_spec_to_gym_space(dm_control_space: specs) -> gym.spaces:
    if isinstance(dm_control_space, specs.BoundedArray):
        low, high = (
            (0, 255)
            if len(dm_control_space.shape) > 1
            else (dm_control_space.minimum, dm_control_space.maximum)
        )
        return spaces.Box(
            low=np.float32(low),
            high=np.float32(high),
            shape=dm_control_space.shape,
            dtype=dm_control_space.dtype
            if len(dm_control_space.shape) > 1
            else np.float32,
        )

    elif isinstance(dm_control_space, specs.Array):
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=np.float32,
        )

    elif isinstance(dm_control_space, dict):
        return spaces.Dict(
            {
                key: convert_spec_to_gym_space(value)
                for key, value in dm_control_space.items()
            }
        )

    else:
        raise ValueError(f"Unsupported DM control space type: {type(dm_control_space)}")


class CathSim(gym.Env):
    def __init__(
        self,
        phantom: str = "phantom3",
        use_contact_forces: bool = False,
        use_force: bool = False,
        use_geom_pos: bool = False,
        return_info: bool = False,
        dm_env=None,
        # delta: float = 0.004,
        delta: float = 0.5,
        # success_reward: float = 10.0,
        success_reward: float = 20.0,
        target: str = "bca",  # 添加target参数，设置默认值
        **kwargs,
    ):
        # self.spec = EnvSpec("cathsim/CathSim-v0", max_episode_steps=300)
        self.spec = EnvSpec("cathsim/CathSim-v0", max_episode_steps=600)
        self.delta = delta
        self.success_reward = success_reward
        self.dm_env = dm_env
        # self.target_x = 0.0085
        # self.target_y = 0.0061
        if self.dm_env is None:
            self._env = make_dm_env(phantom=phantom, **kwargs)
        else:# None!!!
            self._env = self.dm_env

        self.image_size = self._env.task.image_size
        self.camera_matrix = self._env.task.get_camera_matrix(image_size=self.image_size, camera_name="top_camera")
        self.target_3D = self._env._task.target_pos

        self.best_road_mask = cv2.imread("./observation_data/best_road_mask.png", cv2.IMREAD_UNCHANGED)

        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": round(1.0 / self._env.control_timestep()),
        }

        # self.action_space = convert_spec_to_gym_space(
        #     self._env.action_spec(),
        # )
        # self.observation_space = convert_spec_to_gym_space(
        #     self._env.observation_spec(),
        # )

        self.viewer = None
        self.use_contact_forces = use_contact_forces
        self.use_force = use_force
        self.use_geom_pos = use_geom_pos
        self.return_info = return_info

        self.action_space = spaces.Box(
            low=np.float32([-1, -1]),
            high=np.float32([1, 1]),
            shape=[2],
            dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=np.zeros((self.image_size, self.image_size, 4), dtype=np.uint8),
                    high=np.ones((self.image_size, self.image_size, 4), dtype=np.uint8) * 255,
                    shape=[self.image_size, self.image_size, 4],
                    dtype=np.uint8
                ),
                "joint_vel": spaces.Box(
                    low=np.array([-np.inf for i in range(6)], dtype=np.float32),
                    high=np.array([np.inf for i in range(6)], dtype=np.float32),
                    shape=[6],
                    dtype=np.float32
                )
            }
        )

    def get_reward(self, tip_x, tip_y, target_x, target_y):
        tip_pos_2d = [tip_x, tip_y]  # 导丝尖端二维坐标
        target_pos_2d = [target_x, target_y]  # target目标二维坐标

        # 计算二维欧氏距离
        distance = np.linalg.norm(np.array(tip_pos_2d) - np.array(target_pos_2d))
        # print(f"distance:{distance:.4f}")
        # print("self.delta:",self.delta)

        is_successful = distance < self.delta  # 是否达到目标（阈值 delta）
        reward = self.success_reward if is_successful else -distance
        self.success = is_successful  # 更新成功状态

        return reward, is_successful


    def _get_obs(self, timestep):
        obs = timestep.observation
        for key, value in obs.items():
            if value.dtype == np.float64:
                obs[key] = value.astype(np.float32)
        return obs

    def _get_info(self):
        info = dict(
            head_pos=self.head_pos.copy(),
            target_pos=self.target.copy(),
        )

        if self.use_contact_forces:
            info["contact_forces"] = self.contact_forces.copy()

        if self.use_force:
            info["forces"] = self.force.copy()

        if self.use_geom_pos:
            info["geom_pos"] = self.guidewire_geom_pos.copy()
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        timestep = self._env.reset()
        old_obs = self._get_obs(timestep)
        self.step_count = 0  # 重置步骤计数器
        # print("self.target========",self.target)

        # 1. 初始化4通道图像（250x250x4）
        catted_img = np.zeros((self.image_size, self.image_size, 4), dtype=np.uint8)
        # 2. 填充4通道数据（对应目标代码的catted_img）
        # 第0通道：全局灰度图像（从当前pixels的3通道图转换）
        # 假设当前pixels是BGR格式（OpenCV默认），转换为灰度
        catted_img[:, :, 0] = cv2.cvtColor(old_obs["pixels"], cv2.COLOR_BGR2GRAY)

        # 第1通道：血管mask（对应当前的phantom，单通道需压缩维度）
        # observation["phantom"]形状是(250,250,1)，用np.squeeze去除最后一个维度
        catted_img[:, :, 1] = np.squeeze(old_obs["phantom"])

        # 第2通道：最优路径mask（当前观测中无此键，需处理！）
        # 若环境中没有"best_road_mask"，可暂时用0填充，或检查是否遗漏键
        # （若有最优路径数据，替换为对应值，如：observation["best_road"]）
        # catted_img[:, :, 2] = np.zeros((self.image_size, self.image_size), dtype=np.uint8)  # 临时填充0
        catted_img[:, :, 2] = self.best_road_mask

        # 第3通道：导丝mask（对应当前的guidewire，单通道压缩维度）
        catted_img[:, :, 3] = np.squeeze(old_obs["guidewire"])

        # guidewire_mask = np.squeeze(old_obs["guidewire"])
        # # 添加形态学闭运算，连接节段
        # kernel = getStructuringElement(MORPH_ELLIPSE, (1, 1))  # 3x3 椭圆核
        # guidewire_mask = morphologyEx(guidewire_mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填充间隙
        # guidewire_mask = morphologyEx(guidewire_mask, cv2.MORPH_OPEN, kernel)  # 开运算去除噪点
        # # 替换原 mask
        # catted_img[:, :, 3] = guidewire_mask



        # 3. 构建6维joint_vel（对应目标代码的line_obs）
        line_obs = np.zeros(6, dtype=np.float32)

        # 前2维：导丝尖端坐标,转换导丝尖端（tip）的三维坐标到二维
        tip_2d = point2pixel(
            points=old_obs["tip_pos"],
            camera_matrix=self.camera_matrix,
            camera_kwargs=dict(image_size=self.image_size)
        )
        # print(f"导丝尖端二维坐标：{tip_2d}")  # 输出如 [40, 35]（像素坐标）
        # 前2维：导丝尖端坐标（从当前tip_pos提取x、y，假设tip_pos是(x,y,z)）
        # line_obs[0] = old_obs["tip_pos"][2]  # 导丝尖端x
        # line_obs[1] = old_obs["tip_pos"][1]  # 导丝尖端y
        line_obs[0] = tip_2d[0]  # 导丝尖端x
        line_obs[1] = tip_2d[1]  # 导丝尖端y
        # print("old_observation[tip_pos][2]========", old_observation["tip_pos"][2])
        # print("old_observation[tip_pos][1]========", old_observation["tip_pos"][1])

        # 中间2维：目标点坐标，转换目标点坐标的三维坐标到二维
        target_2d = point2pixel(
            points=self.target_3D,
            camera_matrix=self.camera_matrix,
            camera_kwargs=dict(image_size=self.image_size)
        )
        # print(f"target二维坐标：{target_2d}")  # 输出如 [40, 35]（像素坐标）
        line_obs[2] = target_2d[0]  # 目标点x（需替换为实际值）
        line_obs[3] = target_2d[1]  # 目标点y（需替换为实际值）

        # 最后2维：补充维度（目标代码中未明确，可留空或用关节速度的前2维）
        # 示例：用joint_vel的前2个值填充
        line_obs[4] = old_obs["joint_vel"][0] if len(old_obs["joint_vel"]) > 0 else 0.0
        line_obs[5] = old_obs["joint_vel"][1] if len(old_obs["joint_vel"]) > 0 else 0.0

        # 4. 生成新的observation字典
        obs = {
            "pixels": catted_img,
            "joint_vel": line_obs
        }

        return obs, {}

    
    
    def step(self, action: np.ndarray) -> tuple:
        def save_image(image_data: np.ndarray, file_path: str):
            """
            image_data: (H, W, C) 或 (H, W)
            C=1 或 C=3，dtype=uint8
            """
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 去掉尾部长度为1的通道
            if image_data.ndim == 3 and image_data.shape[-1] == 1:
                image_data = image_data.squeeze(-1)

            # 现在 shape 只能是 (H,W) 或 (H,W,3)
            if image_data.ndim not in (2, 3):
                raise ValueError(f"不支持的形状 {image_data.shape}")

            img = Image.fromarray(image_data.astype(np.uint8))
            img.save(file_path)

        # print("self.target========", self.target)
        # print("action : ",action)
        timestep = self._env.step(action)
        old_observation = self._get_obs(timestep)
        self.step_count += 1  # 递增步骤计数器
        # # 若是字典类型，打印键值：单进程时控制台可见；多进程时只在 rank-0 打印
        # if isinstance(old_observation, dict):
        #     print("观测值包含的键:", list(old_observation.keys()))  # ['guidewire', 'joint_pos', 'joint_vel', 'pixels']
        #     for key, value in old_observation.items():
        #     	print(f"{key} 形状: {value.shape if hasattr(value, 'shape') else type(value)}")

        # 1. 初始化4通道图像（250x250x4）
        catted_img = np.zeros((self.image_size, self.image_size, 4), dtype=np.uint8)
        # 2. 填充4通道数据（对应目标代码的catted_img）
        # 第0通道：全局灰度图像（从当前pixels的3通道图转换）
        # 假设当前pixels是BGR格式（OpenCV默认），转换为灰度
        catted_img[:, :, 0] = cv2.cvtColor(old_observation["pixels"], cv2.COLOR_BGR2GRAY)

        # 第1通道：血管mask（对应当前的phantom，单通道需压缩维度）
        # old_observation["phantom"]形状是(250,250,1)，用np.squeeze去除最后一个维度
        catted_img[:, :, 1] = np.squeeze(old_observation["phantom"])

        # 第2通道：最优路径mask（当前观测中无此键，需处理！）
        # 若环境中没有"best_road_mask"，可暂时用0填充，或检查是否遗漏键
        # （若有最优路径数据，替换为对应值，如：old_observation["best_road"]）
        # catted_img[:, :, 2] = np.zeros((self.image_size, self.image_size), dtype=np.uint8)  # 临时填充0
        catted_img[:, :, 2] = self.best_road_mask

        # 第3通道：导丝mask（对应当前的guidewire，单通道压缩维度）
        catted_img[:, :, 3] = np.squeeze(old_observation["guidewire"])

        # guidewire_mask = np.squeeze(old_observation["guidewire"])
        # # 添加形态学闭运算，连接节段
        # kernel = getStructuringElement(MORPH_ELLIPSE, (1, 1))  # 3x3 椭圆核
        # guidewire_mask = morphologyEx(guidewire_mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填充间隙
        # guidewire_mask = morphologyEx(guidewire_mask, cv2.MORPH_OPEN, kernel)  # 开运算去除噪点
        # # 替换原 mask
        # catted_img[:, :, 3] = guidewire_mask

        # 3. 构建6维joint_vel（对应目标代码的line_obs）
        line_obs = np.zeros(6, dtype=np.float32)

        # 前2维：导丝尖端坐标,转换导丝尖端（tip）的三维坐标到二维
        tip_2d = point2pixel(
            points=old_observation["tip_pos"],
            camera_matrix=self.camera_matrix,
            camera_kwargs=dict(image_size=self.image_size)
        )
        # print(f"导丝尖端二维坐标：{tip_2d}")  # 输出如 [40, 35]（像素坐标）
        # 前2维：导丝尖端坐标（从当前tip_pos提取x、y，假设tip_pos是(x,y,z)）
        # line_obs[0] = old_observation["tip_pos"][2]  # 导丝尖端x
        # line_obs[1] = old_observation["tip_pos"][1]  # 导丝尖端y
        line_obs[0] = tip_2d[0]  # 导丝尖端x
        line_obs[1] = tip_2d[1]  # 导丝尖端y
        # print("old_observation[tip_pos][2]========", old_observation["tip_pos"][2])
        # print("old_observation[tip_pos][1]========", old_observation["tip_pos"][1])

        # 中间2维：目标点坐标，转换目标点坐标的三维坐标到二维
        target_2d = point2pixel(
            points=self.target_3D,
            camera_matrix=self.camera_matrix,
            camera_kwargs=dict(image_size=self.image_size)
        )
        # print(f"target二维坐标：{target_2d}")  # 输出如 [40, 35]（像素坐标）
        line_obs[2] = target_2d[0]  # 目标点x（需替换为实际值）
        line_obs[3] = target_2d[1]  # 目标点y（需替换为实际值）

        # 最后2维：补充维度（目标代码中未明确，可留空或用关节速度的前2维）
        # 示例：用joint_vel的前2个值填充
        line_obs[4] = old_observation["joint_vel"][0] if len(old_observation["joint_vel"]) > 0 else 0.0
        line_obs[5] = old_observation["joint_vel"][1] if len(old_observation["joint_vel"]) > 0 else 0.0

        # 4. 生成新的observation字典
        observation = {
            "pixels": catted_img,
            "joint_vel": line_obs
        }


        # # 将导丝尖端二维坐标tip_2d和目标点二维坐标target_2d画在old_observation["pixels"]图中，并保存png图像
        # # 1. 复制原始图像（避免修改原始观测数据）
        # visualized_img = old_observation["pixels"].copy()
        # # 2. 转换坐标为整数像素点
        # tip_pixel = tip_2d
        # target_pixel = target_2d
        # # 3. 绘制导丝尖端（绿色实心圆，半径5）
        # cv2.circle(
        #     img=visualized_img,
        #     center=tip_pixel,
        #     radius=5,
        #     color=(0, 255, 0),  # BGR格式的绿色
        #     thickness=-1  # -1表示填充圆
        # )
        # # 4. 绘制目标点（红色实心圆，半径5）
        # cv2.circle(
        #     img=visualized_img,
        #     center=target_pixel,
        #     radius=5,
        #     color=(0, 0, 255),  # BGR格式的红色
        #     thickness=-1
        # )
        # # 5. 保存绘制后的图像（文件名包含步骤编号，确保唯一）
        # save_path = Path("./tmp_data") / f"step_{self.step_count:04d}_with_markers.png"
        # save_image(visualized_img, str(save_path))
        # print(f"已保存带标记的图像：{save_path}")
        #
        # # 验证新观测值的形状
        # print("新observation的pixels形状：", observation["pixels"].shape)  # 应输出(250,250,4)
        # print("新observation的joint_vel形状：", observation["joint_vel"].shape)  # 应输出(6,)
        #
        # # -------------------------- 1. 准备保存路径（避免文件混乱）--------------------------
        # # 创建专门的保存文件夹（如 "new_observation_data"），不存在则自动创建
        # save_dir = Path("./tmp_data")
        # save_dir.mkdir(exist_ok=True)  # exist_ok=True：文件夹已存在时不报错
        # # -------------------------- 2. 保存 "pixels" 的4个通道（分别保存为单通道图像）--------------------------
        # # 提取new_observation中的4通道图像
        # pixels_4ch = observation["pixels"]  # 形状：(250, 250, 4)
        # # 定义每个通道的含义和文件名（与目标代码的通道对应）
        # channel_info = [
        #     (0, "global_gray", "全局灰度图像"),
        #     (1, "vessel_mask", "血管mask"),
        #     (2, "best_road_mask", "最优路径mask"),
        #     (3, "guidewire_mask", "导丝mask")
        # ]
        # # 循环保存每个通道
        # for channel_idx, name_suffix, desc in channel_info:
        #     # 提取当前通道（形状：(250, 250)）
        #     single_channel = pixels_4ch[:, :, channel_idx]
        #     # 定义保存路径（如 "new_observation_data/pixels_channel0_global_gray.png"）
        #     save_path = save_dir / f"{self.step_count:04d}_pixels_channel{channel_idx}_{name_suffix}.png"
        #     # 保存单通道图像（cv2.imwrite支持单通道uint8格式）
        #     cv2.imwrite(str(save_path), single_channel)
        #     # 打印保存信息
        #     print(f"已保存 {desc}：{save_path}，形状：{single_channel.shape}")
        # # -------------------------- 3. 保存 "joint_vel" 的6个数值（保存为文本文件）--------------------------
        # # 提取6维joint_vel
        # joint_vel = observation["joint_vel"]  # 形状：(6,)
        # # 定义每个索引对应的含义（与目标代码的line_obs对应）
        # joint_vel_info = [
        #     (0, "导丝尖端x坐标", joint_vel[0]),
        #     (1, "导丝尖端y坐标", joint_vel[1]),
        #     (2, "目标点x坐标", joint_vel[2]),
        #     (3, "目标点y坐标", joint_vel[3]),
        #     (4, "补充维度1（原joint_vel[0]）", joint_vel[4]),
        #     (5, "补充维度2（原joint_vel[1]）", joint_vel[5])
        # ]
        # # 保存到文本文件（支持人类可读格式）
        # save_path = save_dir / "joint_vel_values.txt"
        # with open(save_path, "w", encoding="utf-8") as f:
        #     f.write("joint_vel（6维）数值及含义：\n")
        #     f.write("=" * 50 + "\n")
        #     for idx, desc, value in joint_vel_info:
        #         f.write(f"索引{idx}：{desc} = {value:.6f}\n")  # 保留6位小数，确保精度
        #     f.write("=" * 50 + "\n")
        #     f.write(f"完整joint_vel数组（numpy格式）：\n{joint_vel}")
        # # 打印保存信息
        # print(f"\n已保存joint_vel数值：{save_path}，长度：{len(joint_vel)}")



        reward, done = self.get_reward(old_observation["tip_pos"][0],old_observation["tip_pos"][1],
                                                 line_obs[2],line_obs[3])

        # reward = timestep.reward
        # terminated = timestep.last()
        # truncated = False
        info = self._get_info() if self.return_info else {}
        # return observation, reward, terminated, truncated, info

        return observation, reward, done, False, info

    def render_frame(self, image_size=None, camera_id: int = 0):
        image_size = image_size or self.image_size
        img = self._env.physics.render(
            height=image_size, width=image_size, camera_id=camera_id
        )
        return img

    def render(self, mode: str = "rgb_array", image_size: int = None) -> np.ndarray:
        if mode == "rgb_array":
            return self.render_frame(image_size)
        else:
            raise NotImplementedError("Render mode '{}' is not supported.".format(mode))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self._env.close()

    def print_spaces(self):
        print("Observation space:")
        if isinstance(self.observation_space, spaces.Dict):
            for key, value in self.observation_space.spaces.items():
                print("\t", key, value.shape)
        print("Action space:")
        print("\t", self.action_space.shape)

    @property
    def head_pos(self) -> np.ndarray:
        """Get the position of the guidewire tip."""
        return self._env._task.get_head_pos(self.physics)

    @property
    def force(self) -> np.ndarray:
        """The magnitude of the force applied to the aorta."""
        return self._env._task.get_total_force(self.physics)

    @property
    def contact_forces(self) -> np.ndarray:
        """Get the contact forces for each contact point."""
        return self._env._task.get_contact_forces(self.physics, self.image_size)

    @property
    def physics(self):
        """Returns Physics object that is associated with the dm_env."""
        return self._env._physics.copy()

    @property
    def target(self) -> np.ndarray:
        """The target position."""
        return self._env._task.target_pos

    def set_target(self, goal: np.ndarray):
        """Set the target position."""
        self._env._task.set_target(goal)

    @property
    def guidewire_geom_pos(self) -> np.ndarray:
        """The position of the guidewire body geometries. This property is used to determine the shape of the guidewire."""
        return self._env._task.get_guidewire_geom_pos(self.physics)
