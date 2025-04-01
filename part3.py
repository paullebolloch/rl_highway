import pickle
import gymnasium as gym
from matplotlib import pyplot as plt
import highway_env
import time

with open("config_part2.pkl", "rb") as f:
    config_dict = pickle.load(f)

env = gym.make("parking-v0", render_mode="rgb_array")  # Créer l'environnement
env.unwrapped.configure(config_dict)  # Attribuer la configuration à l'environnement
env.reset()  # Réinitialiser l'environnement

# Exécuter 50 actions "neutres"
for _ in range(50):
    time.sleep(0.1)  # 200 ms entre chaque frame
    action = [0.5, 0.0]  # Action neutre dans un espace continu
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()