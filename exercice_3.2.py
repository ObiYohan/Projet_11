# %% [Cellule 3] DQN avec Stable-Baselines3
"""
Étape 3 - Même problème (CartPole-v1), résolu avec Stable-Baselines3.
Comparez la longueur de cette cellule avec les cellules 1 et 2 : c'est
tout l'intérêt d'une bibliothèque de haut niveau.
"""

import gymnasium as gym
from stable_baselines3 import DQN as SB3_DQN  # alias pour ne pas confondre avec notre classe DQN manuelle
from stable_baselines3.common.evaluation import evaluate_policy

# ------------------------------------------------------------------
# 1. Environnement
# ------------------------------------------------------------------
env = gym.make("CartPole-v1")

# ------------------------------------------------------------------
# 2. Instanciation du modèle
# ------------------------------------------------------------------
model = SB3_DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
)

# ------------------------------------------------------------------
# 3. Entraînement
#    total_timesteps = nombre d'interactions (pas de jeu), pas d'épisodes.
#    25 000 est recommandé (mini 5 000).
# ------------------------------------------------------------------
TOTAL_TIMESTEPS = 25_000
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# ------------------------------------------------------------------
# 4. Évaluation sur 100 épisodes
# ------------------------------------------------------------------
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"\n🏆 Entraînement terminé ! ({TOTAL_TIMESTEPS} pas)")
print(f"📊 Récompense moyenne sur 100 épisodes : {mean_reward:.2f} +/- {std_reward:.2f}")

# ------------------------------------------------------------------
# 5. Sauvegarde du modèle (-> dqn_cartpole.zip)
# ------------------------------------------------------------------
model.save("dqn_cartpole")
print("\n💾 Modèle sauvegardé sous dqn_cartpole.zip")

env.close()

# ------------------------------------------------------------------
# (Optionnel) Recharger le modèle sauvegardé
# ------------------------------------------------------------------
# loaded_model = SB3_DQN.load("dqn_cartpole")