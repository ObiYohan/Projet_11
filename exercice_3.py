# %% [Cellule 1] Classes DQN et ReplayBuffer
"""
Étape 1 - Le "cerveau" (DQN) et la "mémoire" (ReplayBuffer) de l'agent.
"""

import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Réseau de neurones qui remplace la Q-table.
    Prend un état en entrée et renvoie une Q-value par action possible.
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        # Les couches sont définies dans __init__
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # Le passage des données à travers les couches se fait dans forward,
        # avec F.relu comme fonction d'activation entre les couches.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # pas d'activation en sortie : ce sont des Q-values brutes


class ReplayBuffer:
    """
    Mémoire des transitions (state, action, reward, next_state, done).
    Un deque(maxlen=capacity) éjecte automatiquement les transitions les
    plus anciennes une fois la capacité atteinte.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition (un tuple) dans le buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Tire un mini-batch aléatoire de transitions."""
        batch = random.sample(self.buffer, batch_size)
        # On "dézippe" la liste de tuples en 5 listes séparées
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # ---- Code instanciant un DQN et affichant son architecture ----
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    policy_net = DQN(state_size, action_size)
    print(f"Taille de l'état : {state_size} | Nombre d'actions : {action_size}\n")
    print("Architecture du réseau DQN :")
    print(policy_net)

    # Petit test rapide de la ReplayBuffer
    buffer = ReplayBuffer(capacity=1000)
    buffer.push([0.1, 0.0, 0.05, -0.1], 1, 1.0, [0.1, 0.1, 0.04, -0.2], False)
    print(f"\nTaille du buffer après un push : {len(buffer)}")


# %% [Cellule 2] Entraînement du DQN manuel
"""
Étape 2 - Boucle d'entraînement complète.

Ne réécrivez pas cette cellule : lisez les commentaires pour comprendre
chaque étape, en particulier optimize_model(), qui contient le coeur de
l'algorithme DQN.
"""

import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# (les classes DQN et ReplayBuffer viennent de la cellule 1)
# from cellule1_classes import DQN, ReplayBuffer  # si exécuté depuis un .py séparé

# ------------------------------------------------------------------
# Hyperparamètres
# ------------------------------------------------------------------
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 500
GAMMA = 0.99                 # facteur d'atténuation du futur
LEARNING_RATE = 1e-3
BUFFER_CAPACITY = 10_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 1_000      # taille mini du buffer avant de commencer à apprendre
TARGET_UPDATE_FREQ = 10      # en épisodes

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Initialisation de l'environnement, des réseaux, de l'optimiseur et du buffer
# ------------------------------------------------------------------
env = gym.make(ENV_NAME)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size).to(device)   # le réseau qu'on entraîne
target_net = DQN(state_size, action_size).to(device)   # la copie "figée" utilisée pour la cible
target_net.load_state_dict(policy_net.state_dict())    # on synchronise au démarrage
target_net.eval()                                       # le target_net n'est jamais entraîné directement

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)


def select_action(state, epsilon):
    """Stratégie epsilon-greedy : exploration vs exploitation."""
    if random.random() < epsilon:
        return env.action_space.sample()  # exploration : action aléatoire
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())  # exploitation : meilleure action connue


def optimize_model():
    """
    Le coeur de l'algorithme DQN : une étape de descente de gradient sur
    un mini-batch tiré du replay buffer.
    """
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None  # pas encore assez de transitions pour apprendre

    # ---- 1. Échantillonnage d'un mini-batch ----
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # ---- 2. Conversion en tenseurs PyTorch ----
    # FloatTensor pour les valeurs continues, LongTensor pour les indices d'action
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)        # shape (batch, 1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)       # shape (batch, 1)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)           # shape (batch, 1)

    # ---- 3. Q-values PRÉDITES par le policy_net pour les actions RÉELLEMENT prises ----
    # policy_net(states) donne une Q-value par action -> shape (batch, action_size)
    # .gather(1, actions) sélectionne, pour chaque ligne, la colonne correspondant
    # à l'action effectivement jouée -> shape (batch, 1)
    current_q_values = policy_net(states).gather(1, actions)

    # ---- 4. Q-value CIBLE, calculée avec le TARGET NETWORK ----
    # C'est le point clé du DQN : on utilise target_net (et non policy_net)
    # pour estimer la meilleure valeur atteignable depuis next_state, ce qui
    # stabilise l'apprentissage (la cible ne "bouge" pas à chaque pas de gradient).
    with torch.no_grad():
        # .max(1)[0] : la plus grande Q-value parmi les actions, pour chaque état du batch
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        # Équation de Bellman : si done=1, il n'y a pas de futur -> le terme disparaît
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    # ---- 5. Calcul de la perte entre prédiction et cible ----
    loss = F.mse_loss(current_q_values, target_q_values)

    # ---- 6. Étape d'optimisation (backpropagation) ----
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ------------------------------------------------------------------
# Boucle d'entraînement principale
# ------------------------------------------------------------------
epsilon = EPSILON_START
episode_rewards = []

for episode in range(NUM_EPISODES):
    state, info = env.reset(seed=episode)
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state

        optimize_model()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    # Le target_net est mis à jour périodiquement en copiant les poids du policy_net
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Progression affichée tous les 50 épisodes
    if episode % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(
            f"Épisode {episode:4d} | récompense : {total_reward:6.1f} | "
            f"moyenne (50 derniers) : {avg_reward:6.1f} | epsilon : {epsilon:.3f}"
        )

env.close()
print("\n🏆 Entraînement terminé !")

# ------------------------------------------------------------------
# Graphique final : récompense par épisode
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Récompense par épisode", alpha=0.6)
# moyenne mobile sur 50 épisodes pour lisser la courbe
if len(episode_rewards) >= 50:
    moving_avg = np.convolve(episode_rewards, np.ones(50) / 50, mode="valid")
    plt.plot(range(49, len(episode_rewards)), moving_avg, label="Moyenne mobile (50)", linewidth=2)
plt.xlabel("Épisode")
plt.ylabel("Récompense totale")
plt.title("DQN manuel sur CartPole-v1 - Récompense par épisode")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("dqn_manuel_rewards.png", dpi=100, bbox_inches="tight")
plt.show()