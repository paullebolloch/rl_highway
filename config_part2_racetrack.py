import pickle

# Configuration for continuous-control Racetrack environment in highway-env
config_dict = {
    # Use kinematic observations (position, velocity, heading)
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "scales": [100, 100, 5, 5, 1, 1]
    },

    # Continuous action space: steering and acceleration
    "action": {
        "type": "ContinuousAction",
        "continuous": {
            "longitudinal": True,   # <- permettre à l’agent de freiner/accélérer
            "lateral":      True
        }
    },

    # Simulation & policy frequencies
    "simulation_frequency": 15,    # Hz: physics simulation step rate
    "policy_frequency": 5,         # Hz: how often the agent acts (must divide simulation_frequency)

    # Episode duration
    "duration": 300,               # max number of policy steps per episode

    # Reward shaping
    "collision_reward": -1.0,
    "lane_centering_cost": 15,
    "action_reward": -0.05,

      # ---- NOUVEAUX TERMES pour vitesse ----
    "reward_speed_range": [0, 30],   # plage de vitesses que l'on récompense
    "speed_reward":        1.0,      # +1 point * forward_speed  
    "reverse_penalty":    -2.0,      # −2 points * backward_speed 

    "controlled_vehicles": 1,
    "other_vehicles": 1,     # continuous reward
    "centering_position": [0.5, 0.5],  # center of the racetrack
    # Rendering settings
    "offscreen_rendering": False,
    "render_agent": True,
    "screen_width": 800,
    "screen_height": 600,
    "scaling": 5.0,
}

# Save to pickle for use by your notebook
with open("config_part2.pkl", "wb") as f:
    pickle.dump(config_dict, f)
