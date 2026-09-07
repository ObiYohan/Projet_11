"""
video_export.py
================
Écrit les frames de plusieurs épisodes réussis dans un unique fichier
.mp4, sauvegardé dans un dossier PERSISTANT du projet (recordings/),
contrairement aux vidéos générées à la volée pour le GUI (app.py) qui
vivent dans le dossier temporaire du système et peuvent disparaître.

C'est le fichier à utiliser pour le livrable "vidéo de 20-30s montrant
une performance réussie".
"""

import os
import time

import imageio.v2 as imageio
import numpy as np

RECORDINGS_DIR = os.environ.get("RL_RECORDINGS_DIR", "recordings")


def build_demo_video(episodes: list[dict], fps: int = 15, output_name: str | None = None) -> dict:
    """
    Concatène les frames de plusieurs épisodes (déjà filtrés comme "réussis"
    par agent.collect_successful_episodes) en une seule vidéo continue.

    Écrit les frames une par une via imageio.get_writer (flux), sans jamais
    charger toutes les frames de tous les épisodes dans un unique tableau
    numpy en mémoire.

    Renvoie le chemin du fichier écrit et la durée réelle obtenue.
    """
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    if output_name is None:
        output_name = f"demo_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(RECORDINGS_DIR, output_name)

    total_frames = 0
    with imageio.get_writer(output_path, fps=fps) as writer:
        for episode in episodes:
            for frame in episode["frames"]:
                writer.append_data(np.asarray(frame))
                total_frames += 1

    duration_seconds = total_frames / fps if fps else 0.0

    return {
        "output_path": output_path,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
    }
