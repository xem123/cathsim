import os
from pathlib import Path

import torch as th
from cathsim.rl.data import RESULTS_PATH
from memory_profiler import profile
from pympler import muppy, summary
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
import csv, os

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}

# class ActionLogger(BaseCallback):
#     def __init__(self, log_dir, freq=1000):
#         super().__init__()
#         self.freq = freq
#         self.path = os.path.join(log_dir, "train_actions.csv")
#         os.makedirs(log_dir, exist_ok=True)
#         with open(self.path, "w") as f:
#             f.write("step,a0,a1\n")
#
#     def _on_step(self) -> bool:
#         if self.num_timesteps % self.freq == 0:
#             obs_key = "obs" if "obs" in self.locals else "new_obs"
#             obs = self.locals[obs_key]
#             action, _ = self.model.predict(obs, deterministic=False)
#             print(f"[{self.num_timesteps}] action = {action[0].tolist()}")
#         return True

class IOCallback(BaseCallback):
    def __init__(self, log_dir, freq=1000):
        super().__init__()
        self.freq = freq
        self.log_file = os.path.join(log_dir, "train_actions.csv")
        with open(self.log_file, "w") as f:
            f.write("step,pixels_shape,guidewire_shape,joint_pos_shape,joint_vel_shape,a0,a1\n")

    def _on_step(self):
        if self.num_timesteps % self.freq == 0:
            obs = self.locals.get("obs") or self.locals.get("new_obs")
            #print("obs['pixels'].shape", obs['pixels'].shape)
            #print("obs['guidewire'].shape", obs['guidewire'].shape)
            #print("obs['joint_pos'].shape", obs['joint_pos'].shape)
            #print("obs['joint_pos']",obs['joint_pos'])
            #print("obs['joint_vel'].shape", obs['joint_vel'].shape)
            #print("obs['joint_vel']",obs['joint_vel'])
            action, _ = self.model.predict(obs, deterministic=False)
            with open(self.log_file, "a") as f:
                f.write(f"{self.num_timesteps},"
                        f"{obs['pixels'].shape},"
                        f"{obs['guidewire'].shape},"
                        # f"{obs['joint_pos'].shape},"
                        # f"{obs['joint_vel'].shape},"
                        f"{obs['joint_pos']},"
                        f"{obs['joint_vel']},"
                        f"{action[0,0]:.4f},{action[0,1]:.4f}\n")
        return True

def generate_experiment_paths(experiment_path: Path = None) -> tuple:
    if experiment_path.is_absolute() is False:
        experiment_path = RESULTS_PATH / experiment_path

    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"
    for directory_path in [experiment_path, model_path, log_path, eval_path]:
        directory_path.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def train(
    algo: str,
    config_name: str = "test",
    target: str = "bca",
    phantom: str = "phantom3",
    trial_name: str = "test2",
    base_path: Path = RESULTS_PATH,
    n_timesteps: int = 600_000,
    n_runs: int = 4,
    evaluate: bool = False,
    n_envs: int = None,
) -> None:
    """Train a model.

    This function trains a model using the specified algorithm and configuration.

    Args:
        algo (str): Algorithm to use. Currently supported: ppo, sac
        config_name (str): The name of the configuration file (see config folder)
        target (str): The target to use. Currently supported: bca, lcca
        phantom (str): The phantom to use.
        trial_name (str): The trial name to use. Used to separate different runs.
        base_path (Path): The base path to use for saving the results.
        n_timesteps (int): Number of timesteps to train for.
        n_runs (int): Number of runs to train.
        evaluate (bool): Flag to evaluate the model after training.
        n_envs (int): Number of environments to use for training. Defaults to half the number of CPU cores.
    """
    from cathsim.rl import Config, make_gym_env # 配置类Config,环境创建函数make_gym_env
    from cathsim.rl.evaluation import evaluate_policy, save_trajectories # 策略评估函数evaluate_policy,轨迹保存函数save_trajectories
    # 创建配置对象config，设置配置名称、试验名称、基础路径和任务参数
    config = Config(
        config_name=config_name,
        trial_name=trial_name,
        base_path=base_path,
        task_kwargs=dict(
            phantom=phantom,
            target=target,
        ),
    )
    print(config)
    # {'algo_kwargs': {'buffer_size': 500000,
    #                  'device': 'cuda',
    #                  'policy': 'MultiInputPolicy',
    #                  'policy_kwargs': {'features_extractor_class': <class 'cathsim.rl.feature_extractors.cnn_extractor.CustomExtractor'>}},
    # 'base_path': PosixPath('/home/xingenming/python_project/cathsim/data/results'),
    # 'config_name': 'full',
    # 'task_kwargs': {'image_size': 80,
    #                 'phantom': 'phantom3',
    #                 'random_init_distance': 0,
    #                 'sample_target': False,
    #                 'target': 'bca',
    #                 'target_from_sites': False,
    #                 'use_pixels': True,
    #                 'use_segment': True},
    # 'trial_name': '2',
    # 'wrapper_kwargs': {'channels_first': False,
    #                    'grayscale': True,
    #                    'time_limit': 300,
    #                    'use_obs': ['pixels','guidewire','joint_pos','joint_vel']}
    # }
    print(f"Training {algo} on {target} using {phantom}")

    experiment_path = config.get_env_path()#通过配置对象获取实验路径
    # 调用generate_experiment_paths函数生成模型保存路径、日志路径和评估路径
    model_path, log_path, eval_path = generate_experiment_paths(experiment_path)

    # 读取env_utils.py中的make_gym_env函数！！！创建强化学习环境，使用指定的配置和环境数量 (如果未指定则使用 CPU 核心数的一半)
    # 返回一个符合Gymnasium API的gym.Env对象（可能是单进程，也可能是向量化多进程），其观测空间、动作空间和reset()/step()接口全部对齐Stable-Baselines3的约定
    # 单进程时是：Monitor(MultiInputImageWrapper(FilterObservation(CathSim(...))))
    # 多进程时是：VecMonitor(SubprocVecEnv([_create_env, _create_env, ..., _create_env]))
    env = make_gym_env(config=config, n_envs=n_envs or os.cpu_count() // 2)

    # 开始多轮训练循环，循环次数由n_runs决定
    for seed in range(n_runs):
        # 根据指定的算法名称从ALGOS字典中获取对应的算法类并初始化模型
        model = ALGOS[algo](
            # 设置模型的环境、日志详细程度、TensorBoard 日志路径和算法特定参数
            env=env,
            verbose=1,
            tensorboard_log=log_path,
            **config.algo_kwargs,
        )

        print(model_path)

        # 如果路径中存在模型就不重新训练，直接加载模型并做评估/保存轨迹
        # ① 加载这个现成的模型，
        # ② 如果需要评估，就再用它跑2个回合生成轨迹并把这2条轨迹以.pkl 文件存到 eval/sac_0/
        # ③ 然后跳过后面的真正训练步骤（continue），避免重复训练。”
        if (model_path / f"{algo}_{seed}.zip").exists():
            print(f"Model {algo}_{seed} already exists, loading model.")
            model = ALGOS[algo].load(model_path / f"{algo}_{seed}.zip")
            print("evaluate???:",evaluate)
            if evaluate:
                # 作用：重新创建一个无监控的环境（避免重复写日志）
                env = make_gym_env(config=config, monitor_wrapper=False)
                # 作用：用已经训练好的模型再跑2个完整 episode，把过程中所有 观测、动作、奖励、信息 记录下来，存成 2 个 .pkl 轨迹文件，方便后续可视化、指标计算、行为克隆等。
                trajectories = evaluate_policy(model, env, n_episodes=2)
                save_trajectories(trajectories, eval_path / f"{algo}_{seed}")
            continue
        # 调用模型的learn方法开始训练
        model.learn(
            # 设置总训练时间步数、显示进度条、不重置时间步数计数器、设置 TensorBoard 日志名称和日志记录间隔
            total_timesteps=n_timesteps,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"{algo}_{seed}",
            log_interval=10,
            callback=IOCallback(log_path),
        )
        # 训练完成后，将模型保存到指定路径
        model.save(model_path / f"{algo}_{seed}")

        if evaluate:
            env = make_gym_env(config=config, monitor_wrapper=False)
            trajectories = evaluate_policy(model, env, n_episodes=2)
            save_trajectories(trajectories, eval_path / f"{algo}_{seed}")
        th.cuda.empty_cache()


if __name__ == "__main__":
    from cathsim.rl import train

    train(
        algo="sac",
        config_name="internal",
        target="bca",
        phantom="phantom3",
        trial_name="test-trial_5",
        base_path=Path.cwd() / "results",
        n_timesteps=1200,
        n_runs=1,
        evaluate=True,
        n_envs=8,
    )
