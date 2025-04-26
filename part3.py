import gymnasium as gym
import pickle
import os
import numpy as np
import highway_env
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from gymnasium.wrappers import RecordVideo


class TensorboardMarkerCallback(BaseCallback):
    """
    A callback that logs custom markers to TensorBoard at key training events.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True    

    def _on_training_start(self) -> None:
        self.logger.record("marker/training_start", 1)

    def _on_rollout_start(self) -> None:
        self.logger.record("marker/rollout_start", 1)

    def _on_rollout_end(self) -> None:
        self.logger.record("marker/rollout_end", 1)

    def _on_training_end(self) -> None:
        self.logger.record("marker/training_end", 1)


class EpisodeRewardCallback(BaseCallback):
    """
    Logs and counts episode rewards at the end of each episode.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_count += 1
                reward = ep_info["r"]
                length = ep_info["l"]
                self.logger.record("episode/reward", reward)
                self.logger.record("episode/length", length)
                print(f"Episode {self.episode_count} ended: reward={reward:.2f}, length={length}")
        return True


def evaluate_random(env, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_part3.pkl")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    video_folder = os.path.join(script_dir, "videos_parking")
    log_folder = os.path.join(script_dir, "parking_ddpg")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Environment factory
    def make_env():
        env = gym.make("parking-v0", render_mode="rgb_array")
        env.unwrapped.configure(config)
        monitor_path = os.path.join(log_folder, "monitor.csv")
        env = Monitor(env, filename=monitor_path)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: ep_id % 10 == 0,
            name_prefix="parking_agent"
        )
        return env

    env = DummyVecEnv([make_env])

    # Action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # Instantiate DDPG
    model = DDPG(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        learning_rate=5e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.8,
        action_noise=action_noise,
        verbose=2,
        tensorboard_log=log_folder,
    )

    # Callbacks
    tb_callback = TensorboardMarkerCallback()
    ep_callback = EpisodeRewardCallback()
    callback = CallbackList([tb_callback, ep_callback])

    # Train
    total_timesteps = int(2e4)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="DDPG_parking"
    )
    print(f"Training completed over {ep_callback.episode_count} episodes.")

    # Save
    model.save(os.path.join(log_folder, "model"))

    # Evaluation
    eval_env = gym.make("parking-v0", render_mode="rgb_array")
    eval_env.unwrapped.configure(config)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Trained agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    random_mean, random_std = evaluate_random(eval_env, n_episodes=20)
    print(f"Random agent: mean_reward={random_mean:.2f} +/- {random_std:.2f}")

    print(f"Performance improvement: {mean_reward - random_mean:.2f} average reward.")
