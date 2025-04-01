import pickle
import gymnasium as gym
from matplotlib import pyplot as plt
import highway_env
import time

with open("config_part3.pkl", "rb") as f:
    config_dict = pickle.load(f)

env = gym.make("highway-v0", render_mode="rgb_array") # Créer l'environnement

env.unwrapped.configure(config_dict) # Attribuer la configuration à l'environnement

env.reset() # Réinitialiser l'environnement

# Exécuter 50 actions IDLE
for _ in range(50):
    time.sleep(0.1)
    action = [0.5, 0.0]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()

