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

from overlay import draw_metrics_overlay

RECORDINGS_DIR = os.environ.get("RL_RECORDINGS_DIR", "recordings")


def build_demo_video(
    episodes: list[dict],
    fps: int = 15,
    output_name: str | None = None,
    overlay_signals: list[str] | None = None,
) -> dict:
    """
    Concatène les frames de plusieurs épisodes (déjà filtrés comme "réussis"
    par agent.collect_successful_episodes) en une seule vidéo continue.

    Écrit les frames une par une via imageio.get_writer (flux), sans jamais
    charger toutes les frames de tous les épisodes dans un unique tableau
    numpy en mémoire.

    overlay_signals : labels de signaux (clés de signals.SIGNAL_COLUMNS) à
    dessiner en surimpression sur chaque frame. None ou [] = pas de
    surimpression, comportement inchangé. Chaque frame est associée à sa
    ligne de step_records via `frame_step_indices` (voir agent.run_episode)
    pour retrouver les valeurs au bon instant, même quand frame_stride > 1
    fait que toutes les frames ne sont pas conservées.

    Renvoie le chemin du fichier écrit et la durée réelle obtenue.
    """
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    if output_name is None:
        output_name = f"demo_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(RECORDINGS_DIR, output_name)

    total_frames = 0
    with imageio.get_writer(output_path, fps=fps) as writer:
        for episode in episodes:
            frame_step_indices = episode.get("frame_step_indices", [])
            step_records = episode.get("step_records", [])

            for i, frame in enumerate(episode["frames"]):
                if overlay_signals:
                    step_idx = frame_step_indices[i] if i < len(frame_step_indices) else None
                    record = step_records[step_idx] if step_idx is not None and step_idx < len(step_records) else None
                    frame_to_write = (
                        draw_metrics_overlay(np.asarray(frame), record, overlay_signals)
                        if record is not None
                        else np.asarray(frame)
                    )
                else:
                    frame_to_write = np.asarray(frame)

                writer.append_data(frame_to_write)
                total_frames += 1

    duration_seconds = total_frames / fps if fps else 0.0

    return {
        "output_path": output_path,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
    }
