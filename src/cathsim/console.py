import argparse as ap  # 导入argparse库，用于解析命令行参数
import os
from pathlib import Path  # 导入Path类，用于路径处理

import torch as th  # 导入PyTorch库，用于深度学习相关操作
from cathsim.rl import Config  # 从cathsim.rl导入Config类，用于加载训练配置
from cathsim.rl.utils import ALGOS, generate_experiment_paths  # 导入算法字典和路径生成函数
import os
os.environ['MUJOCO_GL'] = 'egl'  # 或 'osmesa'
# 导入DM Control或Mujoco
import dm_control
import numpy as np


def cmd_visualize_agent(args=None):
    """可视化训练好的智能体。
    :param args: 命令行参数（默认值为None）
    """
    import argparse as ap  # 再次导入argparse（函数内局部导入，确保作用域正确）
    import cv2  # 导入OpenCV库，用于图像处理和显示
    # from cathsim.utils import make_gym_env  # 导入创建Gym环境的工具函数
    from cathsim.rl.env_utils import make_gym_env
    # from cathsim.visualization import point2pixel  # 导入坐标转换函数（3D点转图像像素）
    from cathsim.dm.visualization import point2pixel
    # from cathsim.wrappers import SingleDict2Array  # 导入自定义包装器（字典观测转数组）
    from cathsim.gym.wrappers import SingleDict2Array
    from stable_baselines3.common.policies import ActorCriticCnnPolicy  # 导入CNN策略类（用于加载BC模型）

    parser = ap.ArgumentParser()  # 创建命令行参数解析器
    # 定义可视化相关参数
    parser.add_argument("--config", type=str, default="full")  # 配置名称（默认"full"）
    parser.add_argument("--base-path", type=str, default="results")  # 结果存储基础路径（默认"results"）
    parser.add_argument("--trial", type=str, default="test-trial")  # 试验名称（默认"1"）
    parser.add_argument("--phantom", type=str, default="phantom3")  # 虚拟模型名称（默认"phantom3"）
    parser.add_argument("--target", type=str, default="bca")  # 目标位置（默认"bca"）
    # parser.add_argument("--seed", type=int, default=None)  # 随机种子（默认None，不指定）
    parser.add_argument("--seed", type=int, default=0)  # 随机种子（默认None，不指定，设置为1则运行训练过程中保存好的sac_1.zip模型）
    parser.add_argument("--save-video", type=bool, default=False)  # 是否保存视频（默认False）
    parser.add_argument("--get_images", type=bool, default=False)  # 是否保存图像（默认False）
    parser.add_argument("--algo", type=str, default="sac")  # 算法名称（默认"sac"）
    parser.add_argument("--visualize-sites", type=bool, default=False)  # 是否可视化关键位置（默认False）
    args = parser.parse_args()  # 解析参数

    algo = args.algo  # 获取算法名称

    if args.save_video:  # 若需保存视频，导入moviepy（视频处理库）
        import moviepy.editor as mpy

    # 构建路径：虚拟模型/目标/配置
    path = Path(f"{args.phantom}/{args.target}/{args.config}")
    config = Config(config_name=args.config)
    # config = Config(args.config)  # 加载配置文件
    # 生成模型、日志、评估路径
    # model_path, log_path, eval_path = generate_experiment_paths(
    #     path, base_path=Path.cwd() / args.base_path / args.trial
    # )
    full_path = Path.cwd() / args.base_path / args.trial / path
    model_path, log_path, eval_path = generate_experiment_paths(full_path)
    print("model_path==",model_path)

    # 确定模型文件路径（根据是否指定seed）
    if args.seed is None:
        model_path = model_path / algo
    else:
        model_path = model_path / (algo + f"_{args.seed}.zip") # model_path是运行训练过程中保存好的sac_0.zip等多个模型
    video_path = model_path.parent.parent / "videos"  # 视频保存路径
    images_path = model_path.parent.parent / "images"  # 图像保存路径

    # 断言模型文件存在，否则报错
    assert model_path.exists(), f"{model_path} 不存在"

    # 更新配置中的任务参数
    # config["task_kwargs"]["target"] = args.target
    config.task_kwargs["target"] = args.target
    # config["task_kwargs"]["phantom"] = args.phantom
    config.task_kwargs["phantom"] = args.phantom
    # config["task_kwargs"]["visualize_sites"] = args.visualize_sites
    config.task_kwargs["visualize_sites"] = args.visualize_sites

    # 加载模型（区分"bc"算法和其他算法）
    if algo == "bc":
        config["wrapper_kwargs"]["channel_first"] = True  # BC算法需要通道优先的图像格式
        env = make_gym_env(config=config)  # 创建Gym环境
        env = SingleDict2Array(env)  # 将字典观测转为数组
        # 初始化BC模型（ActorCriticCnnPolicy）并加载权重
        model = ActorCriticCnnPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: th.finfo(th.float32).max,  # 学习率调度（此处设为最大值，不更新）
        ).load(model_path)
    else:
        env = make_gym_env(config=config)  # 创建Gym环境
        model = ALGOS[algo].load(model_path)  # 从算法字典加载模型

    # 运行10个episode（轮次）可视化
    for episode in range(10):
        numm = 0
        obs = env.reset()  # 重置环境，获取初始观测
        #print("Type of obs:", type(obs))
        #print("Type of obs[0]:", type(obs[0]) if isinstance(obs, tuple) else "not tuple")
        # print("obs=======", obs)
        if isinstance(obs, tuple):
           obs = obs[0]
        #print("obs=======", obs)
        done = False  # 任务完成标志
        frames = []  # 存储视频帧
        segment_frames = []  # 存储分割图像帧
        rewards = []  # 存储每步奖励

        # 单轮episode循环（直到任务完成）
        while not done:
            numm+=1
            if numm>=600:
                break
            print("numm====",numm)
            #action, _states = model.predict(obs)  # 模型根据观测预测动作
            
            action, _ = model.predict(obs)
            # print("action===",action)  # [ 0.9916568  -0.85608953]

            
            # obs, reward, done, info = env.step(action)  # 环境执行动作，返回新状态
            obs, reward, done, truncated, info = env.step(action)  # 环境执行动作，返回新状态
            # input("wait") #用回车控制一步一步走
            # print("obs ======", obs)
            # print("reward ======", reward)
            #print("done ======", done)
            #print("truncated ======", truncated)
            # print("info ======", info)
            rewards.append(reward)  # 记录奖励
            # 渲染图像并处理
            # image = env.render("rgb_array", image_size=480)  # 渲染RGB图像（尺寸480）
            image = env.render()  # 渲染RGB图像（尺寸480）
            image = cv2.resize(image, (480, 480), interpolation=cv2.INTER_CUBIC)
            #锐化
            kernel = np.array([[0, -1,  0],
                   [-1, 5, -1],
                   [0, -1,  0]], dtype=np.float32)
            image = cv2.filter2D(image, -1, kernel)

            frames.append(image)  # 保存帧
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转为BGR格式（OpenCV兼容）
            ## 在图像上绘制目标点（红色实心圆）
            #image = cv2.circle(
             #   image,
                # point2pixel(info["target_pos"], dict(image_size=480)),  # 目标点坐标转像素
            #    radius=10,  # 半径10
             #   color=(0, 0, 255),  # 红色（BGR格式）
            #    thickness=-1,  # 实心圆
            #)
            cv2.imshow("image", image)  # 显示图像
            cv2.waitKey(1)  # 等待1ms，确保图像刷新

        # 若需保存视频
        if args.save_video:
            os.makedirs(video_path, exist_ok=True)  # 创建视频目录（若不存在）
            # 将帧转为视频片段
            frames = [mpy.ImageClip(m).set_duration(0.1) for m in frames]
            clip = mpy.concatenate_videoclips(frames, method="compose")  # 拼接片段
            # 保存视频（帧率60）
            clip.write_videofile(
                (video_path / f"{algo}_{episode}.mp4").as_posix(), fps=60
            )
        # 若需保存图像
        if args.get_images:
            import matplotlib.pyplot as plt  # 导入matplotlib用于保存图像

            os.makedirs(images_path, exist_ok=True)  # 创建图像目录（若不存在）
            print(f"保存图像到 {images_path}")
            # 循环保存每一帧
            for i in range(len(frames)):
                plt.imsave(f"{images_path}/{algo}_{episode}_{i}.png", frames[i])  # 保存RGB图像
                plt.imsave(
                    f"{images_path}/{algo}_{episode}_{i}_segment.png",
                    segment_frames[i],
                    cmap="gray",  # 分割图像用灰度图
                )


def cmd_run_env(args=None):
    """
    运行环境（用于测试或交互）。
    :param args: 命令行参数（默认值为None）
    """
    from argparse import ArgumentParser  # 导入参数解析器
    from cathsim.dm import make_dm_env  # 导入创建DM环境的函数
    from cathsim.dm.utils import launch  # 导入环境启动函数

    ap = ArgumentParser()  # 创建解析器
    # 定义环境运行参数
    ap.add_argument("--interact", type=bool, default=True)  # 是否交互（默认True）
    ap.add_argument("--phantom", default="phantom3", type=str)  # 虚拟模型（默认"phantom3"）
    # ap.add_argument("--phantom", default="artery_0916_move_rotate_x270", type=str)  # 虚拟模型（默认"phantom3"）
    # ap.add_argument("--phantom", default="artery_surface_move_rotate_x270", type=str)  # 虚拟模型（默认"phantom3"）
    ap.add_argument("--target", default="bca", type=str)  # 目标位置（默认"bca"）
    ap.add_argument("--image_size", default=80, type=int)  # 图像尺寸（默认80）
    ap.add_argument("--visualize-target", action="store_true")  # 是否可视化目标（默认不显示）

    args = ap.parse_args(args)  # 解析参数

    # 创建DM环境
    env = make_dm_env(
        phantom=args.phantom,  # 虚拟模型
        use_pixels=True,  # 使用像素观测
        use_segment=True,  # 使用分割图像
        target=args.target,  # 目标位置
        visualize_sites=False,  # 不可视化关键位置
        visualize_target=args.visualize_target,  # 是否可视化目标
    )

    # 1) 打印观测空间
    print("Observation keys:", list(env.observation_spec().keys()))
    print("Observation dtypes/shapes:")
    for k, v in env.observation_spec().items():
        print(f"  {k}: {v.shape} {v.dtype}")
    # 2) 打印动作空间
    print("Action space:", env.action_spec())
    # 3) 打印所有相机
    print("Cameras:", [c.name for c in env._task._arena.cameras]) # ['side']
    # 4) 打印每台相机的完整信息
    print("\n=== 相机详情 ===")
    for c in env._task._arena.cameras:
        print(f"Name : {c.name}")
        print(f"Pos  : {c.pos}")
        print(f"Quat : {c.quat}")
        print(f"FOVy : {getattr(c, 'fovy', 'N/A')}")
        print("-" * 30)

    # 1. 取得底层 physics
    physics = env.physics
    # 2. 打印 top_camera 的 pos 和 quat
    cam_id = physics.model.name2id('top_camera', 'camera')
    for cam_id in range(physics.model.ncam):
        name   = physics.model.id2name(cam_id, 'camera')
        pos    = physics.model.cam_pos[cam_id]
        quat   = physics.model.cam_quat[cam_id]
        print(f'ID={cam_id:2d}  name={name:20s}  pos={pos}  quat={quat}')
    pos  = physics.model.cam_pos[cam_id]
    quat = physics.model.cam_quat[cam_id]
    print('top_camera pos :', pos)
    print('top_camera quat:', quat)

    launch(env)  # 启动环境（进入交互或运行模式）


def cmd_train(args=None):
    from cathsim.rl import train  # 从rl模块导入train函数

    parser = ap.ArgumentParser()  # 创建参数解析器
    # 定义训练相关参数
    parser.add_argument("-a", "--algo", type=str, default="sac")  # 算法（默认"sac"）
    parser.add_argument("-c", "--config", type=str, default="full")  # 配置（默认"test"）
    parser.add_argument("-t", "--target", type=str, default="bca")  # 目标（默认"bca"）
    parser.add_argument("-p", "--phantom", type=str, default="phantom3")  # 虚拟模型（默认"phantom3"）
    # parser.add_argument("-p", "--phantom", type=str, default="artery0002_0829")  # 虚拟模型（默认"phantom3"）
    parser.add_argument("--trial-name", type=str, default="test-trial")  # 试验名称（默认"test-trial"）
    parser.add_argument("--base-path", type=Path, default=Path.cwd() / "results")  # 结果路径（默认当前目录/results）
    parser.add_argument("--n-runs", type=int, default=1)  # 运行次数（默认1）
    parser.add_argument("--n-timesteps", type=int, default=int(6e5))  # 训练步数（默认60万）
    parser.add_argument("-e", action="store_false")  # evaluate评估标志（默认True，-e禁用）
    # parser.add_argument("-e", action="store_true", default=True)
    args = parser.parse_args()  # 解析参数


    # 调用train函数启动训练
    train(
        algo=args.algo,  # 算法'sac'
        config_name=args.config,  # 配置名称'test'
        target=args.target,  # 目标位置'bca'
        phantom=args.phantom,  # 虚拟模型'phantom3'
        trial_name=args.trial_name,  # 试验名称'test-trial'
        base_path=Path.cwd() / args.base_path,  # 结果保存路径Path('results')
        n_timesteps=args.n_timesteps,  # 训练总步数
        n_runs=args.n_runs,  # 重复运行次数
        evaluate=args.e,  # 是否训练后评估
    )
