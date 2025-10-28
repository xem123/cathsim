from stable_baselines3.common.callbacks import BaseCallback
import torch

class ActionPrintCallback(BaseCallback):
    def __init__(self, freq: int = 1000):
        super().__init__()
        self.freq = freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            # 取第 0 号环境的当前观测
            obs = self.locals["obs"]
            # 模型预测
            action, _ = self.model.predict(obs, deterministic=True)
            # action 形状 (n_envs, 2) -> 打印第 0 个
            print(f"[step {self.num_timesteps}] action = {action[0].tolist()}")
        return True