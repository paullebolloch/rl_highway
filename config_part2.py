import pickle

config_dict = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "scales": [100, 100, 5, 5, 1, 1]
    },
    "action": {
        "type": "ContinuousAction"  # <-- Changement ici
    },
    "simulation_frequency": 15,    # Hz : rapidité de la simulation
    "policy_frequency": 5,         # fréquence des actions (doit diviser simulation_frequency)
    "duration": 40,                # nombre d'étapes max par épisode
    "vehicles_count": 8,           # autres véhicules sur l’intersection
    "initial_vehicle_count": 1,    # ton agent uniquement au début
    "controlled_vehicles": 1,      # nombre d’agents RL (souvent 1)
    "collision_reward": -1.0,
    "reward_speed_range": [4, 6],
    "reward_type": "dense",        # plus stable qu’un reward sparse
    "offscreen_rendering": False,
    "render_agent": True,
    "screen_width": 800,
    "screen_height": 600,
    "scaling": 5.0,
    "center_at_agent": True        # suivi dynamique
}

# Sauvegarde dans un fichier .pkl
with open("config_part3.pkl", "wb") as f:
    pickle.dump(config_dict, f)