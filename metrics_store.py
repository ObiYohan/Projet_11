"""
metrics_store.py
=================
Persistance simple des métriques de performance (une ligne par épisode
joué). Utilise un CSV en append-only plutôt qu'une base de données pour
rester simple à déployer sur un Hugging Face Space.

Point de vigilance RAM : load_history() lit le fichier par CHUNKS
(pandas chunksize) plutôt que d'un bloc, pour éviter de saturer la RAM
si l'historique grandit beaucoup au fil des parties jouées sur le Space.
"""

import csv
import os
from collections import Counter
from typing import Iterator

import pandas as pd

DATA_DIR = os.environ.get("RL_DATA_DIR", "data")
METRICS_FILE = os.path.join(DATA_DIR, "episodes.csv")
FIELDNAMES = ["episode_id", "total_reward", "steps", "action_counts_json"]

CHUNK_SIZE = 500  # nombre de lignes lues à la fois par load_history()


def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_episode(total_reward: float, steps: int, actions: list) -> int:
    """
    Ajoute une ligne pour un épisode qui vient d'être joué.
    Renvoie l'episode_id attribué (compteur incrémental basé sur le
    nombre de lignes déjà présentes).
    """
    _ensure_file()

    action_counts = dict(Counter(str(a) for a in actions))

    # on compte les lignes existantes pour attribuer un id (évite de tout
    # charger en mémoire juste pour ça)
    with open(METRICS_FILE, "r") as f:
        episode_id = sum(1 for _ in f) - 1  # -1 pour l'entête déjà lue au premier passage
    episode_id = max(episode_id, 0)

    import json

    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(
            {
                "episode_id": episode_id,
                "total_reward": total_reward,
                "steps": steps,
                "action_counts_json": json.dumps(action_counts),
            }
        )

    return episode_id


def load_history_chunks() -> Iterator[pd.DataFrame]:
    """
    Générateur qui lit l'historique par petits blocs (CHUNK_SIZE lignes),
    plutôt qu'un seul gros DataFrame en mémoire. À utiliser si l'historique
    devient volumineux.
    """
    _ensure_file()
    for chunk in pd.read_csv(METRICS_FILE, chunksize=CHUNK_SIZE):
        yield chunk


def load_history() -> pd.DataFrame:
    """
    Charge tout l'historique d'un coup. Pratique pour le dashboard tant que
    le nombre de parties reste raisonnable (quelques milliers de lignes,
    un CSV de ce type reste de l'ordre de quelques centaines de Ko).
    Pour un historique massif, préférer load_history_chunks() et agréger
    au fil de l'eau.
    """
    chunks = list(load_history_chunks())
    if not chunks:
        return pd.DataFrame(columns=FIELDNAMES)
    return pd.concat(chunks, ignore_index=True)


def summary_stats() -> dict:
    """Statistiques agrégées, calculées chunk par chunk pour limiter la RAM."""
    count = 0
    total = 0.0
    total_sq = 0.0
    best = float("-inf")
    worst = float("inf")

    for chunk in load_history_chunks():
        rewards = chunk["total_reward"].to_numpy()
        count += len(rewards)
        total += rewards.sum()
        total_sq += (rewards**2).sum()
        if len(rewards):
            best = max(best, rewards.max())
            worst = min(worst, rewards.min())

    if count == 0:
        return {"n_episodes": 0, "mean_reward": None, "std_reward": None, "best": None, "worst": None}

    mean = total / count
    variance = max(total_sq / count - mean**2, 0.0)
    std = variance**0.5

    return {
        "n_episodes": count,
        "mean_reward": float(mean),
        "std_reward": float(std),
        "best": float(best),
        "worst": float(worst),
    }
