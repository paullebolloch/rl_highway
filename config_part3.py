import pickle
import gymnasium as gym

config_dict = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "ContinuousAction"
    },
    "vehicles_count": 2,
    "duration": 100,
    "collision_reward": -1,
    "reward_type": "dense", # récompense à chaque étape
    "goal_reward": 1.0,
    "distance_reward": 0.1,
    "simulation_frequency": 15,   # plus bas = plus lent
    "policy_frequency": 5, 
    "screen_width": 600,
    "screen_height": 300,
    "scaling": 5.5,
    "render_agent": True,
    "center_at_agent": False,
    "centering_position": [0.5, 0.5],
    "offscreen_rendering": False
}

with open("config_part3.pkl", "wb") as f:
    pickle.dump(config_dict, f)

#env = gym.make("parking-v0", render_mode="rgb_array")
#env.unwrapped.configure(config_dict)
#print(env.reset())
