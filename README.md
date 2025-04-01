# 🚗 Projet Reinforcement Learning – HighwayEnv & ParkingEnv

Projet de la Mention IA – CentraleSupélec  
Encadré par Hédi Hadiji — Avril 2025
Groupe: Quentin Lemboulas, Théo Michel, Paul Le Bolloch

## 🎯 Objectif du projet

Implémenter trois agents RL dans des environnements de conduite simulée :

1. **Tâche 1 – Implémentation DQN avec environnement discret spécifié** 
2. **Tâche 2 – Implémentation avec actions continues**
3. **Tâche 3 – Utilisation de StableBaselines dans un nouvel environnement**

---

---

## ✅ Résumé des configurations

### 🔹 `config_part1.pkl` – **Tâche 1**
- Environnement : `highway-fast-v0`
- Observation : `OccupancyGrid`
- Action : `DiscreteMetaAction`
- Implémentation DQN

### 🔹 `config_part3.pkl` – **Tâche 2**
- Environnement : `highway-v0`
- Observation : `Kinematics` (normalisée)
- Action : `ContinuousAction`
- Implémentation d’un algo pour actions continues

### 🔹 `config_part2.pkl` – **Tâche 3**
- Environnement : `parking-v0`
- Observation : `KinematicsGoal`
- Action : `ContinuousAction`
- Apprentissage via `Stable-Baselines3`
