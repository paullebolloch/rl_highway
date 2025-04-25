import pickle
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import highway_env

# Load environment configuration
with open("config_part1.pkl", "rb") as f:
    config = pickle.load(f)

# Create and configure the environment with rgb_array render mode
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.unwrapped.configure(config)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def make_env():
    env_local = gym.make("highway-fast-v0", render_mode="rgb_array")
    env_local.unwrapped.configure(config)
    return env_local

# Hyperparameters
gamma = 0.99
batch_size = 64
buffer_capacity = 10000
update_target_every = 50  # steps before target network update
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 1e-4       # decrease per step
learning_rate = 1e-3
num_episodes = 200
max_steps = int(config.get("duration", 60) * config.get("policy_frequency", 1))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, state, action, reward, done, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, done, next_state)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)
    def __len__(self):
        return len(self.memory)

# Neural Network for Q-values
class Net(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# DQN Agent
class DQNAgent:
    def __init__(self, obs_shape, n_actions):
        self.obs_dim = int(np.prod(obs_shape))
        self.n_actions = n_actions
        self.q_net = Net(self.obs_dim, 128, n_actions)
        self.target_net = Net(self.obs_dim, 128, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.epsilon = epsilon_start
        self.losses = []

    def get_action(self, state, use_epsilon=True):
        if use_epsilon and random.random() < self.epsilon:
            return env.action_space.sample()
        state_v = torch.tensor(state, dtype=torch.float32).view(1, -1)
        q_vals = self.q_net(state_v)
        return int(torch.argmax(q_vals, dim=1).item())

    def append_and_learn(self, state, action, reward, done, next_state):
        self.buffer.push(state, action, reward, done, next_state)
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        states_arr = np.stack(states).astype(np.float32)
        next_states_arr = np.stack(next_states).astype(np.float32)
        states_v = torch.from_numpy(states_arr).view(batch_size, -1)
        next_states_v = torch.from_numpy(next_states_arr).view(batch_size, -1)
        actions_v = torch.tensor(actions, dtype=torch.int64).view(batch_size, 1)
        rewards_v = torch.tensor(rewards, dtype=torch.float32)
        dones_v = torch.tensor(dones, dtype=torch.float32)
        q_values = self.q_net(states_v).gather(1, actions_v).squeeze(1)
        next_q_values = self.target_net(next_states_v).max(1)[0]
        expected_q = rewards_v + gamma * next_q_values * (1 - dones_v)
        loss = nn.MSELoss()(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        if self.steps_done % update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = max(epsilon_end, self.epsilon - epsilon_decay)
        self.steps_done += 1
        return loss.item()

# Training function
def train_agent(agent, episodes):
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0.0
        for t in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated
            agent.append_and_learn(state, action, reward, done, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
    return rewards

# Instantiate and train agent
# Determine observation shape from a sample state, not from observation_space
sample_obs, _ = env.reset()
obs_shape = np.array(sample_obs, dtype=np.float32).shape
agent = DQNAgent(obs_shape, env.action_space.n)
all_rewards = train_agent(agent, num_episodes)

# Plot performance
plt.figure()
plt.title("Episode Rewards")
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
plt.figure()
plt.title("Loss during training")
plt.plot(agent.losses)
plt.xlabel("Training Step")
plt.ylabel("MSE Loss")
plt.show()

# Save trained model
model_path = "dqn_highway_model.pth"
torch.save(agent.q_net.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Function to load and evaluate a saved model with frame display for rgb_array

def load_and_evaluate(path, episodes=5, render_pause=0.05):
    eval_env = make_env()
    # Get correct observation shape for the new env instance
    sample_obs, _ = eval_env.reset()
    obs_shape = np.array(sample_obs, dtype=np.float32).shape
    eval_agent = DQNAgent(obs_shape, eval_env.action_space.n)
    eval_agent.q_net.load_state_dict(torch.load(path))
    eval_agent.q_net.eval()
    plt.ion()
    fig, ax = plt.subplots()
    for ep in range(episodes):
        state, _ = eval_env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0.0
        while not done:
            frame = eval_env.render()
            ax.clear()
            ax.imshow(frame)
            ax.axis('off')
            plt.pause(render_pause)
            action = eval_agent.get_action(state, use_epsilon=False)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = np.array(next_state, dtype=np.float32)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.2f}")
    plt.ioff()
    eval_env.close()

# Example: Evaluate and display
load_and_evaluate(model_path, episodes=3)
