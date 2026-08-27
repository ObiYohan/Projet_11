import random
import gymnasium as gym
import numpy as np

# 1. Création de l'environnement FrozenLake
# is_slippery=False rend le lac non-glissant pour simplifier l'apprentissage au début
env = gym.make("FrozenLake-v1", is_slippery=False)
# env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# 2. Initialisation de la Q-table (16 états x 4 actions) remplie de zéros
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# 3. Hyperparamètres de l'algorithme
total_episodes = 2000  # Nombre de parties à jouer
learning_rate = 0.8  # Taux d'apprentissage (Alpha)
discount_rate = 0.95  # Facteur d'atténuation du futur (Gamma)

# Paramètres pour la stratégie Exploration/Exploitation (Epsilon-Greedy)
epsilon = 1.0  # Taux d'exploration initial (100% au début)
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005  # Vitesse de réduction de l'exploration

# 4. Cycle d'apprentissage
for episode in range(total_episodes):
    state, info = env.reset()
    done = False

    while not done:
        # Étape A : Choisir une action (Exploration vs Exploitation)
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            # Exploitation : on prend la meilleure action de la Q-table
            action = np.argmax(q_table[state, :])
        else:
            # Exploration : on choisit une action au hasard
            action = env.action_space.sample()

        # Étape B : Exécuter l'action dans le jeu
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Étape C : Mettre à jour la Q-table à l'aide de l'équation de Bellman
        # Q(s,a) = Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # nouvelle_valeur = ancienne_valeur + lr * (recompense + gamma * max_q_futur - ancienne_valeur)
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward
            + discount_rate * np.max(q_table[new_state, :])
            - q_table[state, action]
        )

        # Le nouvel état devient l'état actuel
        state = new_state

    # Réduire epsilon pour faire de moins en moins d'exploration au fil du temps
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
        -decay_rate * episode
    )

print("🏆 Entraînement terminé !\n")
print("📊 Extrait de la Q-Table finale (Lignes = États, Colonnes = Actions) :")
print("Actions : [0: Gauche, 1: Bas, 2: Droite, 3: Haut]\n")

# Affichage des lignes de la table
for s in range (len(q_table)):
    print(f"Case {s+1:2d} :  {np.round(q_table[s], 3)}")


# 5. Cycle d'évaluation
total_episodes = 100
win_nb = 0

for episode in range(total_episodes):
    state, info = env.reset()
    done = False

    while not done:
        # Étape A : Choisir une action
        action = np.argmax(q_table[state, :])

        # Étape B : Exécuter l'action dans le jeu
        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated : win_nb += 1
        done = terminated or truncated

        # Le nouvel état devient l'état actuel
        state = new_state

print("🏆 Evaluation terminé !\n")
print(f"Nombre de victoires : {win_nb}")

env.close()