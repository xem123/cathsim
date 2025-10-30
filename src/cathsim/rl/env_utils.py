from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import cv2 as cv2
from PIL import Image
import os

# 过滤环境观测中的特定键值，只保留 filter_keys 中指定的观测项。
def apply_filter_observation(env: gym.Env, filter_keys: Optional[list]) -> gym.Env:
    if filter_keys:
        from gymnasium import wrappers

        env = wrappers.FilterObservation(env, filter_keys=filter_keys)
    return env

# 如果启用了像素观测（use_pixels=True），则对图像进行预处理
def apply_multi_input_image_wrapper(env: gym.Env, options: Dict[str, Any]) -> gym.Env:
    # 若 use_pixels=True（即 "pixels" 键存在且是图片），再包一层 MultiInputImageWrapper
    if options.get("use_pixels", False):
        from cathsim.gym.wrappers import MultiInputImageWrapper

        env = MultiInputImageWrapper(
            env,
            grayscale=options.get("grayscale", False),
            image_key=options.get("image_key", "pixels"),
            keep_dim=options.get("keep_dim", True),# 会保留通道维 (80,80,1) 而不是 (80,80)
            channel_first=options.get("channel_first", False),
        )
    return env

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

# 创建并配置完整的 Gym 环境，支持多进程并行和环境监控。
"""
解析配置参数
创建基础环境 cathsim/CathSim-v0
应用观测过滤（只保留指定的观测项）
应用图像预处理（如果使用像素观测）
支持多进程并行（通过 SubprocVecEnv）
添加监控包装器（用于 TensorBoard 可视化）
"""
def make_gym_env(
        config: dict = {}, # 外部传进来的超参字典；若空则全部走默认
        n_envs: int = 1, # 要并行起多少个 MuJoCo 实例；1 表示单进程
        monitor_wrapper: bool = True # 是否在最外层包 Monitor/VecMonitor 用来写 TensorBoard
) -> gym.Env: # 返回的永远是 gym.Wrapper 或其派生类
    """Makes a gym environment given a configuration. This is a wrapper for the creation of environment and basic wrappers

    Args:
        config: dict:  (Default value = {}) The configuration dictionary
        n_envs: int:  (Default value = 1) The number of environments to create
        monitor_wrapper: bool:  (Default value = True) Whether or not to use the monitor wrapper

    Returns:
        gym.Env: The environment

    """
    """创建一个Gym环境，支持多进程和环境包装"""
    # | 层级               | 类/函数                               | 作用（一句话说清）                                                                       |
    # | ----------------- | ---------------------------------- | ------------------------------------------------------------------------------- |
    # | **0 原始层**       | dm\_control `composer.Environment` | 真正的 MuJoCo 物理仿真器，返回 `timestep`                                                  |
    # | **1 Gym Adapter** | `CathSim`                          | 把 dm\_env 包装成 Gym 接口：`reset()`→`obs dict`, `step()`→`(obs, reward, done, info)` |
    # 0 原始层：只会说 dm_control 方言，相当于“裸 MuJoCo”。
    # 1 Gym Adapter：把方言翻译成 标准 Gym 普通话，让任何 Gym-based RL 库（SB3, RLlib…）都能直接听懂

    # | **2 观测过滤**      | `FilterObservation`                | 把 `obs dict` 里不想要的键删掉，网络永远看不到                                                   |
    # | **3 图像处理**      | `MultiInputImageWrapper`           | 把像素图灰度化、缩放、调 HWC/CHW，统一数值范围                                                     |
    # | **4 单进程监控**    | `Monitor`                          | 记录每个 episode 的 `ep_len`, `ep_rew`, `ep_time` → csv + TensorBoard                |
    # | **5 向量化**       | `SubprocVecEnv`                    | 开多个子进程，每个子进程内部跑 1-4 层；父进程自动把 8 份结果堆叠成 batch                                     |
    # | **6 多进程监控**    | `VecMonitor`                       | 在多进程环境里继续统计 episode 信息并写日志                                                      |
    # | **7 最外层**        | 整个 `make_gym_env()` 返回值        | 供 SB3 算法直接使用：`model = SAC(env=make_gym_env(...))`                               |
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    wrapper_kwargs = config.wrapper_kwargs or {}   # 观测包装器相关参数
    task_kwargs = config.task_kwargs or {}   # 直接传给底层 dm_env 的任务参数

    # 作用：给外层 make_gym_env 提供一份干净、已预先包好必要 Wrapper 的单个 Gym 环境
    # 解释：就是“单条生产线”，每次调用都产出一个已过滤、已图像处理、已监控的单个环境。
    def _create_env() -> gym.Env:
        from cathsim.gym.envs import CathSim

        # 运行gym.make函数创建基础环境，进入 CathSim.__init__()
        # • 读 phantom3.xml → 构造 dm_control composer.Task → Navigate
        # • 设置 delta=0.004, success_reward=10.0, dense_reward=True
        # • 调用 dm_control composer.Environment() 创建底层 dm_env
        # • 再包一层 gym.Wrapper → CathSim 的 Gym 接口
        env = gym.make("cathsim/CathSim-v0", **task_kwargs)

        # # 2. 打印观测空间（observation_space）：模型输入的结构
        # print("观测空间（observation_space）：")
        # print(env.observation_space)
        # # 若观测空间是字典格式（如之前分析的 Dict 类型），可进一步打印每个子空间
        # if isinstance(env.observation_space, gym.spaces.Dict):
        #     for key, space in env.observation_space.spaces.items():
        #         print(f"  {key}: {space}")
        # print("-" * 50)
        #
        # # 3. 打印动作空间（action_space）：模型输出的范围
        # print("动作空间（action_space）：")
        # print(env.action_space)
        # # 若动作空间是 Box 类型，可查看其范围和维度
        # if isinstance(env.action_space, gym.spaces.Box):
        #     print(f"  维度：{env.action_space.shape}")
        #     print(f"  最小值：{env.action_space.low}")
        #     print(f"  最大值：{env.action_space.high}")
        # print("-" * 100)


        # 观测过滤：只保留想要的键
        env = apply_filter_observation(
            env, filter_keys=wrapper_kwargs.get("use_obs", [])
        )

        # # 2. 打印观测空间（observation_space）：模型输入的结构
        # print("观测空间（observation_space）：")
        # print(env.observation_space)
        # # 若观测空间是字典格式（如之前分析的 Dict 类型），可进一步打印每个子空间
        # if isinstance(env.observation_space, gym.spaces.Dict):
        #     for key, space in env.observation_space.spaces.items():
        #         print(f"  {key}: {space}")
        # print("-" * 50)
        #
        # # 3. 打印动作空间（action_space）：模型输出的范围
        # print("动作空间（action_space）：")
        # print(env.action_space)
        # # 若动作空间是 Box 类型，可查看其范围和维度
        # if isinstance(env.action_space, gym.spaces.Box):
        #     print(f"  维度：{env.action_space.shape}")
        #     print(f"  最小值：{env.action_space.low}")
        #     print(f"  最大值：{env.action_space.high}")
        # print("-" * 50)

        # 图像预处理包装器（仅当使用像素）
        env = apply_multi_input_image_wrapper(
            env,
            options={
                # 灰度化：cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                "grayscale": wrapper_kwargs.get("grayscale", True),
                # 只处理指定 image_key，不影响其他观测
                "image_key": wrapper_kwargs.get("image_key", "pixels"),
                # keep_dim=True 会保留通道维 (80,80,1) 而不是 (80,80)
                "keep_dim": wrapper_kwargs.get("keep_dim", True),
                # 维度顺序：HWC → HWC（默认） 或 HWC → CHW（BC 时）
                "channel_first": wrapper_kwargs.get("channel_first", False),
            },
        )
        return env # 返回已包好 Filter + Image 的单环境，env 此时是 原始 dm_env → Gym 的 Adapter


    # 根据 n_envs 决定单/多进程
    if n_envs > 1:
        envs = [_create_env for _ in range(n_envs)]
        env = SubprocVecEnv(envs)  # 多进程环境
        # SubprocVecEnv函数作用：
        # – 每个子进程独立 import MuJoCo，有自己的 OpenGL context。
        # – 父进程通过 multiprocessing.Pipe 发送 step(action)，接收 (obs,rew,done,info)。
        # – 观测自动堆叠成 batch：(n_envs, 80,80,1) 等。
    else:
        env = _create_env()  # 单进程环境
      

    # 添加监控包装器（记录奖励、步数等）
    # Monitor：写 ep_rew, ep_len, ep_time 到 logs/ 下的 csv + TensorBoard events。
    # VecMonitor：多进程下把每个子进程的 episode 统计量合并后写日志。
    if monitor_wrapper:
        env = Monitor(env) if n_envs == 1 else VecMonitor(env)

    # # 新增：测试打印观测值
    # print("\n=== 环境观测结构测试 ===")
    # obs = env.reset()
    #
    # # print("初始观测值:")
    # # print(obs)
    # print("观测值类型:", type(obs))#  <class 'dict'>
    #
    # # 若是字典类型，打印键值：单进程时控制台可见；多进程时只在 rank-0 打印
    # if isinstance(obs, dict):
    #     print("观测值包含的键:", list(obs.keys()))  # ['guidewire', 'joint_pos', 'joint_vel', 'pixels']
    #     for key, value in obs.items():
    #         print(f"{key} 形状: {value.shape if hasattr(value, 'shape') else type(value)}")
    #         # guidewire 形状: (32, 80, 80, 1),joint_pos 形状: (32, 168),joint_vel 形状: (32, 168),pixels 形状: (32, 80, 80, 3)
    #         # print("value====",value)
    #         print("像素值范围:", value.min(), value.max(), value.dtype)
    #         # print("value.dtype==",value.dtype)
    # # 保存pixels和guidewire为图像
    # if 'pixels' in obs and 'guidewire' in obs:
    #     pixels = obs['pixels']
    #     guidewire = obs['guidewire']
    #     # 保存pixels
    #     for i in range(pixels.shape[0]):
    #         pixel_image = pixels[i]
    #         pixel_path = f"/home/xingenming/Downloads/cathsim/results/pixel_{i}.png"
    #         save_image(pixel_image, pixel_path)
    #     # 保存guidewire
    #     for i in range(guidewire.shape[0]):
    #         guidewire_image = guidewire[i]
    #         guidewire_path = f"/home/xingenming/Downloads/cathsim/results/guidwire_{i}.png"
    #         save_image(guidewire_image, guidewire_path)
    # print("save png success----------------")


    # # 测试一步动作后的观测
    # if n_envs == 1:  # 单环境才测试
    #     action = env.action_space.sample()  # 随机采样一个动作
    #    # obs, reward, done, info = env.step(action)
    #     bs, reward, done, truncated, info = env.step(action)
    #     print("\n执行一步动作后的观测值:")
    #     # print(obs)
    # 返回一个符合Gymnasium API的gym.Env对象（可能是单进程，也可能是向量化多进程），其观测空间、动作空间和reset()/step()接口全部对齐Stable-Baselines3的约定
    # 单进程时是：Monitor(MultiInputImageWrapper(FilterObservation(CathSim(...))))
    # 多进程时是：VecMonitor(SubprocVecEnv([_create_env, _create_env, ..., _create_env]))
    return env


if __name__ == "__main__":
    from cathsim.rl import Config

    # config = Config()
    # env = make_gym_env(config, n_envs=1, monitor_wrapper=True)
    config = {
        "task_kwargs": {
            "dense_reward": True,
            "phantom": "phantom3",
            "target": "bca",
            "use_pixels": True,  # 使用像素观测
        },
        "wrapper_kwargs": {
            "use_obs": ["pixels", "guidewire"],  # 只保留像素和导丝状态
            "grayscale": True,  # 转为灰度图
            "channel_first": False,  # 使用 HWC 格式
        }
    }

    env = make_gym_env(config, n_envs=4)  # 创建4个并行环境
    print(env)
