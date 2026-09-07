"""
agent.py
========
Toute la logique liée au modèle RL vit ici, et UNIQUEMENT ici (ou dans
metrics_store.py). Ce module est importé par api.py (backend), jamais
directement par app.py (frontend Gradio) : le frontend ne fait que des
appels HTTP vers l'API, il n'a aucune connaissance du modèle.

Le modèle par défaut est un PPO entraîné sur LunarLanderContinuous-v3
(action_space continu, 2 floats : poussée moteur principal + moteur
latéral). Pour revenir à un DQN discret (LunarLander-v3), changez
ENV_NAME/MODEL_PATH/MODEL_ALGO ci-dessous, ou via les variables
d'environnement RL_ENV_NAME / RL_MODEL_PATH / RL_MODEL_ALGO — le reste du
code est agnostique à l'algorithme grâce à l'API commune de
Stable-Baselines3 (predict()).
"""

import os
from typing import Generator

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO

# ------------------------------------------------------------------
# Configuration du modèle
# ------------------------------------------------------------------
ENV_NAME = os.environ.get("RL_ENV_NAME", "LunarLanderContinuous-v3")
MODEL_PATH = os.environ.get("RL_MODEL_PATH", "model/ppo_lunarlander_continuous.zip")
MODEL_ALGO = os.environ.get("RL_MODEL_ALGO", "PPO")  # "DQN" ou "PPO"

_ALGO_CLASSES = {"DQN": DQN, "PPO": PPO}


class Agent:
    """
    Encapsule le modèle entraîné et l'environnement associé.
    Chargé UNE SEULE FOIS au démarrage de l'API (voir api.py, lifespan),
    pas à chaque requête, pour éviter de recharger les poids en RAM
    inutilement.
    """

    def __init__(self, env_name: str = ENV_NAME, model_path: str = MODEL_PATH, algo: str = MODEL_ALGO):
        self.env_name = env_name
        algo_cls = _ALGO_CLASSES.get(algo, DQN)

        if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                f"Modèle introuvable à '{model_path}'. "
                f"Entraînez-le d'abord (voir train_and_export.py) et placez le "
                f".zip dans le dossier model/."
            )

        self.model = algo_cls.load(model_path)

    # ------------------------------------------------------------------
    # Endpoint /play : état -> action
    # ------------------------------------------------------------------
    def predict_action(self, state: list[float], deterministic: bool = True):
        """
        Reçoit un état brut (liste de floats) et renvoie l'action prédite.
        Ne crée PAS de nouvel environnement à chaque appel : c'est une
        simple passe forward dans le réseau, donc peu coûteux même appelé
        souvent.
        """
        obs = np.array(state, dtype=np.float32)
        action, _states = self.model.predict(obs, deterministic=deterministic)

        # Un DQN renvoie un entier (numpy scalar), un PPO continu renvoie
        # un array de floats -> on normalise le format de sortie en JSON-friendly
        if isinstance(action, np.ndarray):
            action_out = action.tolist()
        else:
            action_out = int(action)

        return action_out

    @staticmethod
    def _describe_action(action) -> str:
        """
        Convertit une action brute en étiquette lisible, utilisée pour le
        dashboard "décisions par phase de vol" (metrics_store / /metrics/decisions).

        - Action discrète (DQN, un entier) : l'étiquette est l'action elle-même.
        - Action continue (PPO, liste de floats) : compter les valeurs EXACTES
          n'aurait aucun sens (deux floats sont presque toujours différents
          d'un pas à l'autre). On regroupe donc chaque dimension en quelques
          catégories (fort/faible/off, gauche/centre/droite) pour obtenir un
          histogramme lisible malgré l'espace continu.
        """
        if isinstance(action, list):
            # LunarLanderContinuous-v3 : action[0] = moteur principal (-1 à 1),
            # action[1] = moteur latéral (-1 = droite, +1 = gauche)
            main, lateral = action[0], action[1]

            if main > 0.5:
                main_label = "principal:fort"
            elif main > 0.0:
                main_label = "principal:faible"
            else:
                main_label = "principal:off"

            if lateral > 0.3:
                lat_label = "lateral:gauche"
            elif lateral < -0.3:
                lat_label = "lateral:droite"
            else:
                lat_label = "lateral:centre"

            return f"{main_label} | {lat_label}"

        return str(action)  # action discrète : l'entier lui-même suffit

    # ------------------------------------------------------------------
    # Endpoint /simulate : joue un épisode complet côté serveur
    # ------------------------------------------------------------------
    def run_episode(
        self,
        deterministic: bool = True,
        max_steps: int = 1000,
        frame_stride: int = 3,
        env_kwargs: dict | None = None,
    ) -> dict:
        """
        Joue un épisode complet et renvoie :
        - la récompense totale et le nombre de pas
        - la liste des actions jouées
        - un sous-échantillon de frames RGB pour l'animation du GUI
        - un historique pas-à-pas (step_records) : phase, action (label +
          composantes séparées), récompense ET observation complète à
          chaque pas, pour visualiser les réactions du modèle dans le temps
          (dashboard "actions vs observations")

        frame_stride > 1 réduit le nombre de frames conservées (donc la RAM
        utilisée) sans casser la fluidité perçue de l'animation.

        env_kwargs : paramètres physiques de LunarLander (gravity, enable_wind,
        wind_power, turbulence_power) transmis tels quels à gym.make(). None
        ou {} = valeurs par défaut de l'environnement.
        """
        env = gym.make(self.env_name, render_mode="rgb_array", **(env_kwargs or {}))
        obs, info = env.reset()

        total_reward = 0.0
        actions_taken = []
        frames = []
        frame_step_indices = []  # frames[i] correspond à step_records[frame_step_indices[i]]
        step_records = []  # historique pas-à-pas : observation + action + récompense, dans le temps

        for step in range(max_steps):
            action = self.predict_action(obs.tolist(), deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            actions_taken.append(action)

            if step % frame_stride == 0:
                frames.append(env.render())  # tableau numpy RGB, gardé en mémoire seulement ici
                frame_step_indices.append(step)

            # On catégorise grossièrement la "circonstance" pour le dashboard :
            # altitude (obs[1]) comme proxy simple de la phase de vol
            altitude = float(obs[1])
            if altitude > 1.0:
                phase = "haute_altitude"
            elif altitude > 0.3:
                phase = "approche"
            else:
                phase = "atterrissage"

            # Action continue (PPO) : 2 floats -> colonnes séparées pour le
            # graphique. Action discrète (DQN) : un entier -> sa propre colonne.
            # Les deux représentations coexistent dans le record ; celle qui
            # ne s'applique pas au modèle courant reste à None (filtrée côté GUI).
            if isinstance(action, list):
                action_main = float(action[0])
                action_lateral = float(action[1]) if len(action) > 1 else None
                action_discrete = None
            else:
                action_main = None
                action_lateral = None
                action_discrete = int(action)

            # Espace d'observation LunarLander (8 dimensions), nommé pour
            # être lisible dans les logs / le tableau / le graphique, plutôt
            # que de laisser un vecteur brut opaque.
            step_records.append({
                "step": step,
                "phase": phase,
                "action": self._describe_action(action),
                "action_raw": action,
                "action_main": action_main,
                "action_lateral": action_lateral,
                "action_discrete": action_discrete,
                "reward": float(reward),
                "altitude": altitude,           # alias de obs_y, gardé pour compat avec /metrics/decisions historique
                "observation": obs.tolist(),    # vecteur brut complet (8 floats), pour usages génériques
                "obs_x": float(obs[0]),
                "obs_y": float(obs[1]),
                "obs_vx": float(obs[2]),
                "obs_vy": float(obs[3]),
                "obs_angle": float(obs[4]),
                "obs_angular_velocity": float(obs[5]),
                "obs_leg1_contact": float(obs[6]),
                "obs_leg2_contact": float(obs[7]),
            })

            obs = next_obs
            if terminated or truncated:
                break

        env.close()

        return {
            "total_reward": total_reward,
            "steps": step + 1,
            "actions": actions_taken,
            "frames": frames,          # liste de np.ndarray (H, W, 3)
            "frame_step_indices": frame_step_indices,  # frames[i] <-> step_records[frame_step_indices[i]]
            "step_records": step_records,
        }

    def iter_episode_frames(
        self, deterministic: bool = True, max_steps: int = 1000, frame_stride: int = 3
    ) -> Generator[np.ndarray, None, None]:
        """
        Version GÉNÉRATRICE de run_episode, pour les cas où on veut streamer
        les frames au fur et à mesure plutôt que de tout garder en RAM
        (utile si max_steps est grand ou si plusieurs parties tournent en
        parallèle). Rendu disponible pour un usage futur type
        StreamingResponse côté API.
        """
        env = gym.make(self.env_name, render_mode="rgb_array")
        obs, info = env.reset()
        for step in range(max_steps):
            action = self.predict_action(obs.tolist(), deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if step % frame_stride == 0:
                yield env.render()
            if terminated or truncated:
                break
        env.close()

    # ------------------------------------------------------------------
    # Endpoint /record_demo : rejoue des épisodes jusqu'à en avoir assez
    # de RÉUSSIS pour constituer une vidéo de démo (consigne : 20-30s
    # montrant une performance réussie).
    # ------------------------------------------------------------------
    def collect_successful_episodes(
        self,
        num_episodes: int = 3,
        reward_threshold: float = 200.0,
        max_attempts: int = 20,
        deterministic: bool = True,
        max_steps: int = 1000,
        env_kwargs: dict | None = None,
    ) -> dict:
        """
        Rejoue des épisodes (frame_stride=1 : on garde TOUTES les frames,
        contrairement à /simulate, car chaque frame compte pour la durée
        de la vidéo finale) jusqu'à en avoir accumulé `num_episodes` dont
        la récompense totale dépasse `reward_threshold` (= "performance
        réussie"), ou jusqu'à épuiser `max_attempts` tentatives.

        On ne garde QUE les frames des épisodes réussis dans le résultat :
        les tentatives ratées ne sont jouées que pour être filtrées, elles
        n'apparaissent jamais dans la vidéo finale.
        """
        successful = []
        all_attempt_rewards = []
        attempts = 0

        while len(successful) < num_episodes and attempts < max_attempts:
            attempts += 1
            result = self.run_episode(
                deterministic=deterministic, max_steps=max_steps, frame_stride=1, env_kwargs=env_kwargs
            )
            all_attempt_rewards.append(result["total_reward"])

            if result["total_reward"] >= reward_threshold:
                successful.append(result)

        return {
            "episodes": successful,               # liste de résultats run_episode (avec frames)
            "attempts": attempts,
            "all_attempt_rewards": all_attempt_rewards,
            "target_reached": len(successful) >= num_episodes,
        }
