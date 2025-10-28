# import numpy as np
# from pathlib import Path
#
# import torch as th
#
# from cathsim.rl.utils import (
#     process_transitions,
#     generate_experiment_paths,
#     make_vec_env,
# )
#
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common import policies
#
# from imitation.algorithms import bc
#
#
# class CnnPolicy(policies.ActorCriticCnnPolicy):
#     """A CNN policy for behavioral clonning."""
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#
# if __name__ == "__main__":
#     model_path, log_path, eval_path = generate_experiment_paths("trial_3")
#
#     rng = np.random.default_rng(0)
#     env = make_vec_env(wrapper_kwargs=dict(time_limit=2000))
#
#     trial_path = Path.cwd() / "rl" / "expert" / "trial_2"
#     transitions = process_transitions(trial_path)
#
#     policy = ActorCriticPolicy(
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         lr_schedule=lambda _: th.finfo(th.float32).max,
#     )
#
#     bc_trainer = bc.BC(
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         policy=policy,
#         demonstrations=transitions,
#         rng=rng,
#     )
#
#     print("Training a policy using Behavior Cloning")
#     bc_trainer.train(n_epochs=500)
#     bc_trainer.save_policy(model_path / "bc_baseline.zip")
#
#     rewards, lengths = evaluate_policy(
#         bc_trainer.policy,
#         env,
#         n_eval_episodes=30,
#         return_episode_rewards=True,
#     )
#
#     print(f"Reward after training: {np.mean(rewards)}")
#     print(f"Lengths: {np.mean(lengths)}")

import numpy as np
# 导入NumPy库，并将其重命名为np，NumPy用于进行高效的数值计算和数组操作。
from pathlib import Path
# 从pathlib模块导入Path类，用于处理文件和目录路径。
import torch as th
# 导入PyTorch库，并将其重命名为th，PyTorch是一个深度学习框架，用于构建和训练神经网络。
from cathsim.rl.utils import (
    process_transitions,
    generate_experiment_paths,
    make_vec_env,
)
# 从cathsim.rl.utils模块导入三个函数：
# process_transitions：用于处理过渡数据。
# generate_experiment_paths：用于生成实验所需的路径。
# make_vec_env：用于创建向量环境。

from stable_baselines3.common.evaluation import evaluate_policy
# 从stable_baselines3.common.evaluation模块导入evaluate_policy函数，用于评估策略的性能。
from stable_baselines3.common.policies import ActorCriticPolicy
# 从stable_baselines3.common.policies模块导入ActorCriticPolicy类，这是一个用于策略网络的基类。
from stable_baselines3.common import policies
# 从stable_baselines3.common模块导入policies子模块，该子模块包含各种策略相关的类和函数。
from imitation.algorithms import bc
# 从imitation.algorithms模块导入bc子模块，bc代表行为克隆（Behavior Cloning），用于实现行为克隆算法。

class CnnPolicy(policies.ActorCriticCnnPolicy):
    """A CNN policy for behavioral clonning."""
    # 定义一个名为CnnPolicy的类，继承自policies.ActorCriticCnnPolicy类，用于行为克隆的卷积神经网络策略。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 类的构造函数，调用父类的构造函数来初始化对象。

if __name__ == "__main__":
    # 当脚本作为主程序运行时，执行以下代码。
    model_path, log_path, eval_path = generate_experiment_paths("trial_3")
    # 调用generate_experiment_paths函数，生成名为"trial_3"的实验所需的路径，分别为模型保存路径、日志路径和评估路径。

    rng = np.random.default_rng(0)
    # 创建一个NumPy的随机数生成器，种子为0，确保实验的可重复性。

    env = make_vec_env(wrapper_kwargs=dict(time_limit=2000))
    # 调用make_vec_env函数创建向量环境，设置包装器的参数time_limit为2000，表示每个回合的最大步数。

    trial_path = Path.cwd() / "rl" / "expert" / "trial_2"
    # 使用Path类构建专家数据所在的路径，该路径为当前工作目录下的rl/expert/trial_2。

    transitions = process_transitions(trial_path)
    # 调用process_transitions函数处理专家数据，将处理后的过渡数据存储在transitions变量中。

    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: th.finfo(th.float32).max,
    )
    # 创建一个ActorCriticPolicy对象，使用环境的观测空间和动作空间进行初始化，并设置学习率调度函数，这里将学习率设置为浮点数的最大值。

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        demonstrations=transitions,
        rng=rng,
    )
    # 创建一个行为克隆训练器bc.BC对象，传入环境的观测空间、动作空间、策略对象、专家过渡数据和随机数生成器。

    print("Training a policy using Behavior Cloning")
    # 打印提示信息，表示正在使用行为克隆算法训练策略。

    bc_trainer.train(n_epochs=500)
    # 调用训练器的train方法，训练策略500个轮次。

    bc_trainer.save_policy(model_path / "bc_baseline.zip")
    # 调用训练器的save_policy方法，将训练好的策略保存到模型保存路径下的bc_baseline.zip文件中。

    rewards, lengths = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=30,
        return_episode_rewards=True,
    )
    # 调用evaluate_policy函数评估训练好的策略，在环境中运行30个回合，并返回每个回合的奖励和回合长度。

    print(f"Reward after training: {np.mean(rewards)}")
    # 打印训练后策略的平均奖励。

    print(f"Lengths: {np.mean(lengths)}")
    # 打印训练后策略的平均回合长度。