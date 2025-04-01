import pickle
import gymnasium as gym
from matplotlib import pyplot as plt
import highway_env

with open("config_part1.pkl", "rb") as f:
    config_dict = pickle.load(f)

env = gym.make("highway-fast-v0", render_mode="rgb_array") # Créer l'environnement

env.unwrapped.configure(config_dict) # Attribuer la configuration à l'environnement

env.reset() # Réinitialiser l'environnement

# Exécuter 50 actions IDLE
for _ in range(50):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()

