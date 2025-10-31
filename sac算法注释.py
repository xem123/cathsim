from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

SelfSAC = TypeVar("SelfSAC", bound="SAC") # 定义一个类型变量 SelfSAC，用于在类方法里返回 自身类型（泛型约束）


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    软演员 - 评论家（SAC）
    带随机演员的离线最大熵深度强化学习，
    本实现借鉴了以下项目的代码：
    原始实现（https://github.com/haarnoja/sac）、
    OpenAI Spinning Up（https://github.com/openai/spinningup）、
    softlearning 仓库（https://github.com/rail-berkeley/softlearning/）
    以及 Stable Baselines（https://github.com/hill-a/stable-baselines）
    论文：https://arxiv.org/abs/1801.01290
    SAC 简介：https://spinningup.openai.com/en/latest/algorithms/sac.html

    注意：如https://github.com/hill-a/stable-baselines/issues/270中讨论的，我们使用双 Q 目标而非价值目标

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance

    :param policy: 要使用的策略模型（MlpPolicy、CnnPolicy 等）
    :param env: 用于学习的环境（如果已在 Gym 中注册，可传入字符串）
    :param learning_rate: Adam 优化器的学习率，
    所有网络（Q 值网络、演员网络和价值函数）将使用相同的学习率，
    它可以是当前剩余进度（从 1 到 0）的函数
    :param buffer_size: 回放缓冲区的大小
    :param learning_starts: 开始学习前，模型需要收集多少步的转换数据
    :param batch_size: 每次梯度更新的小批量数据大小
    :param tau: 软更新系数（“Polyak 更新”，取值在 0 到 1 之间）
    :param gamma: 折扣因子
    :param train_freq: 每 “train_freq” 步更新一次模型。也可传入一个包含频率和单位的元组，
    例如 “(5, "step")” 或 “(2, "episode")”
    :param gradient_steps: 每次滚动后执行的梯度步数（见 “train_freq”），
    设置为 “-1” 表示执行与环境中滚动步数相同的梯度步数
    :param action_noise: 动作噪声类型（默认无），这对困难的探索问题可能有帮助。
    参见 common.noise 了解不同的动作噪声类型
    :param replay_buffer_class: 要使用的回放缓冲区类（例如 “HerReplayBuffer”），
    如果为 “None”，将自动选择
    :param replay_buffer_kwargs: 创建回放缓冲区时要传入的关键字参数
    :param optimize_memory_usage: 启用回放缓冲区的内存高效变体，
    但会增加复杂度，
    详见https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: 当 n_step > 1 时，更新 Q 值网络时使用 n 步回报（配合 NStepReplayBuffer）
    :param ent_coef: 熵正则化系数（相当于原始 SAC 论文中的奖励尺度的倒数），
    用于控制探索与利用的权衡，
    设置为 “auto” 可自动学习（“auto_0.1” 表示使用 0.1 作为初始值）
    :param target_update_interval: 每 “target_network_update_freq” 个梯度步更新一次目标网络
    :param target_entropy: 学习 “ent_coef” 时的目标熵（当 “ent_coef = 'auto'” 时）
    :param use_sde: 是否使用广义状态依赖探索（gSDE）替代动作噪声探索（默认：False）
    :param sde_sample_freq: 使用 gSDE 时，每 n 步采样一个新的噪声矩阵，
    默认：-1（仅在滚动开始时采样）
    :param use_sde_at_warmup: 在预热阶段（学习开始前），是否使用 gSDE 替代均匀采样
    :param stats_window_size: 滚动日志的窗口大小，指定用于计算平均成功率、平均 episode 长度和平均奖励的 episode 数量
    :param tensorboard_log: tensorboard 的日志位置（如果为 None，则不记录日志）
    :param policy_kwargs: 创建策略时要传入的额外参数，详见：ref:sac_policies
    :param verbose: 详细程度：0 表示无输出，1 表示信息性消息（如使用的设备或包装器），2 表示调试消息
    :param seed: 伪随机生成器的种子
    :param device: 运行代码的设备（cpu、cuda 等），
    设置为 “auto” 时，将尽可能使用 GPU
    :param _init_setup_model: 是否在实例创建时构建网络
    """

    # 类变量，把字符串别名映射到具体策略类，方便用户用字符串指定策略。
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    policy: SACPolicy # 完整策略对象
    actor: Actor # 策略网络（输出动作）
    critic: ContinuousCritic # 双 Q 网络（当前）
    critic_target: ContinuousCritic # 双 Q 网络（目标）

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy  # 目标熵（学习熵系数时使用）
        self.log_ent_coef = None  # 类型：Optional[th.Tensor]（熵系数的对数）
        # 熵系数 / 熵温度（奖励尺度的倒数）
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval  # 目标网络更新间隔
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None  # 熵系数优化器

        if _init_setup_model:
            self._setup_model()  # 初始化模型,建立网络结构

    def _setup_model(self) -> None:
        super()._setup_model()  # 调用父类的_setup_model方法
        self._create_aliases()  # 创建别名
        # 收集 critic 和 critic_target 中所有 BatchNorm 的 running_mean/var 参数，做软更新时，需要把这些统计量也同步给目标网络
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])  # 从评论家网络获取批量归一化的统计量（滑动均值/方差）
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])  # 从目标评论家网络获取批量归一化的统计量
        # 学习熵系数时使用目标熵：如果 target_entropy 设为 "auto"，则把它设成「动作维度负数」（这是SAC论文建议的启发式：-dim(A) 作为目标策略熵）；否则转成 float
        if self.target_entropy == "auto":
            # 必要时自动设置目标熵
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # 类型：ignore
        else:
            # 强制转换类型（对于非预期字符串会抛出错误）
            self.target_entropy = float(self.target_entropy)

        # 熵系数或熵可以自动学习，参见论文https://arxiv.org/abs/1812.05905中“Maximum Entropy RL的自动熵调整”部分
        # 作用：实现「自动调整温度 α」功能，见 SAC 论文第 5 节
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # 学习熵系数时的默认初始值
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "熵系数的初始值必须大于0"

            # 注意：我们优化的是熵系数的对数，这与论文略有不同，如https://github.com/rail-berkeley/softlearning/issues/37中讨论的那样
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)  # 初始化熵系数的对数并开启梯度
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))  # 初始化熵系数优化器
        else:
            # 强制转换为float（对于格式错误的字符串会抛出错误）
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    # 把策略内部的 actor、critic、critic_target 提出别名，方便后续直接访问
    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        训练，执行指定步数的梯度更新。
        1、采样回放池：replay_data = self.replay_buffer.sample(...)
        从经验池随机取一个批次数据。
        2、动作与对数概率：actions_pi, log_prob = self.actor.action_log_prob(...)
        用当前策略网络采样动作并计算对数概率。
        3、熵系数更新（自动温度调整）：启用自动调整，计算 ent_coef_loss 并优化 log_ent_coef。
        4、目标 Q 值计算：用目标网络计算 next_q_values，加入熵项，得到 target_q_values。
        5、Critic 损失：计算 MSE 损失并反向传播更新两个 Q 网络。
        6、Actor 损失：actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        最大化 Q 值同时保持高熵，反向传播更新策略网络。
        7、软更新目标网络： target_update_interval 步，用 polyak_update 把当前网络参数软拷贝到目标网络。
        """
        # 切换到训练模式（影响批量归一化/ dropout），确保使用正确的 running stats 与 dropout mask。
        self.policy.set_training_mode(True)
        # 更新优化器的学习率，支持学习率随训练进度线性/余弦衰减
        optimizers = [self.actor.optimizer, self.critic.optimizer]  # 演员和评论家的优化器
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]  # 加入熵系数优化器

        # 根据学习率调度更新学习率
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []  # 熵系数损失和熵系数列表
        actor_losses, critic_losses = [], []  # 演员损失和评论家损失列表

        # 训练主循环：执行指定次数的梯度更新。
        for gradient_step in range(gradient_steps):
            # 1、从经验池采样一个批次；获取 (s,a,r,s′,done) 数据，用于计算目标。
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # 类型：ignore[union-attr]
            # 对于n步回放，折扣因子为gamma**n_steps（无提前终止时），如果是 n_step buffer，则用自带的折扣
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # 如果使用 gSDE，则每一步重置噪声矩阵，保证探索噪声的随机性。
            if self.use_sde:
                self.actor.reset_noise()  # 重置演员的噪声（如果使用广义状态依赖探索）

            # 2、把当前状态喂给策略网络，得到动作及其对数概率，并 reshape 成列向量，在后面算熵正则化目标时要用 log_prob
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)  # Actor网络输出动作和动作的对数概率
            log_prob = log_prob.reshape(-1, 1)  # 重塑对数概率形状

            # 熵系数 α 的损失与优化，实现自动温度调整；记录值用于日志
            ent_coef_loss = None
            # 如果 α 是可学习的：用 exp(log_ent_coef) 得到正数 α，计算 dual gradient descent loss：让策略熵逼近目标熵。
            # 如果 α 是不可学习的：α 是固定标量。
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # 重要：将变量从计算图中分离，避免被其他损失影响（参见https://github.com/rail-berkeley/softlearning/issues/60）
                ent_coef = th.exp(self.log_ent_coef.detach())  # 熵系数=exp(熵系数的对数)（detach避免梯度传播）
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()  # 熵系数损失
                ent_coef_losses.append(ent_coef_loss.item())  # 记录熵系数损失
            else:
                ent_coef = self.ent_coef_tensor  # 使用固定熵系数

            ent_coefs.append(ent_coef.item())  # 记录当前熵系数

            # 反向传播并更新 log_ent_coef =alpha =  优化熵系数（也称为熵温度或论文中的alpha）
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()  # 清空熵系数优化器梯度
                ent_coef_loss.backward()  # 反向传播计算梯度
                self.ent_coef_optimizer.step()  # 更新熵系数

            # 目标 Q 值计算（双重 clipped）,这是 SAC 的 off-policy target，用于训练 Q 网络。
            """用当前策略在 s′ 采样 a′ 与 log_prob；
               用 target-critic 网络计算两个 Q-target，取最小值（抑制过高估计）；
               减去熵正则项 α log π；
               组合成 Bellman 目标：r + γ(Q − α log π)。
            """
            with th.no_grad():  # 不计算梯度（目标网络不更新）
                # 根据策略选择动作next_actions！！！！！！！
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)  # 下一个状态的动作和对数概率
                # 计算下一个Q值：所有目标评论家的最小值
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions),
                                       dim=1)  # 拼接目标评论家的输出
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)  # 取最小值（抑制过估计）
                # 加入熵项
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # TD误差 + 熵项
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values  # 目标Q值

            # 5、Critic 损失
            # 使用回放缓冲区中的真实动作（真实动作如何获取？？直接从经验回放池（ReplayBuffer）里采样得到），把真实动作喂给两个 Q 网络，得到当前 Q，计算与 target 的 MSE；然后反向传播更新 critic 网络
            # 作用：训练 Q 网络逼近最优 soft Q 函数。
            current_q_values = self.critic(replay_data.observations, replay_data.actions)  # 评论家输出的当前Q值，其中replay_data.actions是真实动作，直接从经验回放池（ReplayBuffer）里采样得到的。
            # 计算评论家损失
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
                )  # 均方误差损失（乘以0.5是为了梯度计算方便）
            assert isinstance(critic_loss, th.Tensor)  # 类型检查
            critic_losses.append(critic_loss.item())  # 记录评论家损失
            # 优化评论家
            self.critic.optimizer.zero_grad()  # 清空评论家优化器梯度
            critic_loss.backward()  # 反向传播计算梯度
            self.critic.optimizer.step()  # 更新评论家网络参数

            # 6、Actor 损失，训练策略网络（actor）
            # 用当前策略采样的动作再进 Q 网络，取最小 Q；Actor 的 loss 是 α log π − Q。
            # 梯度上升 Q 同时让策略熵不过低。
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)  # 评论家对演员动作的Q值估计
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)  # 取最小值
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()  # 演员损失
            actor_losses.append(actor_loss.item())  # 记录演员损失
            # 优化策略网络（actor）
            self.actor.optimizer.zero_grad()  # 清空演员优化器梯度
            actor_loss.backward()  # 反向传播计算梯度
            self.actor.optimizer.step()  # 更新演员网络参数

            # 7、软更新目标网络：每隔一定步数，用 τ 系数软更新目标 critic；同时同步 BatchNorm 统计量。
            # 作用：让目标网络缓慢跟踪主网络，稳定训练
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)  # Polyak软更新目标评论家网络
                # 复制滑动统计量（参见GH issue #996）
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)  # 复制批量归一化的滑动统计量

        self._n_updates += gradient_steps  # 更新总更新步数

        # 记录日志
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    # 公共训练入口：露给用户的 learn() 方法，把所有参数透传给父类，由 OffPolicyAlgorithm 统一调度环境交互与 train() 的调用。
    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:# 指定保存模型时「不保存」的键名，这些网络权重已经包含在 policy 中，避免重复保存
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    # 告诉 SB3 的保存工具：需要序列化哪些 state_dict 与张量变量（log_ent_coef 或 ent_coef_tensor），保证保存/加载后恢复完整的训练状态
    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
