import os
import tempfile
import time

import gradio as gr
import numpy as np
import pandas as pd
import requests

from api import app as fastapi_app 

# En interne, le GUI appelle l'API sur le même process (boucle locale).
# PORT doit rester cohérent avec le port passé à uvicorn.run() en bas de ce fichier.
PORT = int(os.environ.get("PORT", 7860))
API_BASE_URL = os.environ.get("RL_API_BASE_URL", f"http://127.0.0.1:{PORT}")


# ------------------------------------------------------------------
# Callbacks GUI : "voir une partie jouée"
# ------------------------------------------------------------------
def play_episode(max_steps: int, frame_stride: int, progress=gr.Progress()):
    """
    Appelle POST /simulate et transforme les frames reçues en une liste
    d'images pour gr.Gallery/gr.Image (Gradio gère l'animation via un
    slider ou en jouant la galerie image par image).
    """
    progress(0, desc="Simulation en cours côté API...")
    resp = requests.post(
        f"{API_BASE_URL}/simulate",
        json={
            "deterministic": True,
            "max_steps": int(max_steps),
            "frame_stride": int(frame_stride),
            "include_frames": True,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    # Décodage base64 -> images affichables. On le fait ici (frontend),
    # pas de logique métier, juste de l'affichage.
    import base64
    import io
    from PIL import Image

    images = []
    for b64 in data["frames_base64"]:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        images.append(img)

    summary = (
        f"**Épisode #{data['episode_id']}** — "
        f"Récompense totale : **{data['total_reward']:.2f}** — "
        f"Durée : {data['steps']} pas"
    )
    return images, summary


def play_episode_as_video(max_steps: int, frame_stride: int):
    """
    Variante : construit un court fichier vidéo (mp4) à partir des frames,
    pour un rendu "animation d'atterrissage" plus fluide qu'une galerie.
    Utilise imageio en générateur d'écriture (pas de gros tableau numpy
    unique en mémoire) pour rester léger en RAM.
    """
    resp = requests.post(
        f"{API_BASE_URL}/simulate",
        json={
            "deterministic": True,
            "max_steps": int(max_steps),
            "frame_stride": int(frame_stride),
            "include_frames": True,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    import base64
    import io

    import imageio.v2 as imageio
    from PIL import Image

    video_path = os.path.join(tempfile.gettempdir(), f"episode_{data['episode_id']}_{int(time.time())}.mp4")
    with imageio.get_writer(video_path, fps=15) as writer:
        for b64 in data["frames_base64"]:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            writer.append_data(np.array(img))  # écrite frame par frame, pas de buffer global

    summary = (
        f"**Épisode #{data['episode_id']}** — "
        f"Récompense totale : **{data['total_reward']:.2f}** — "
        f"Durée : {data['steps']} pas"
    )
    return video_path, summary


# ------------------------------------------------------------------
# Callback GUI : enregistrement de la vidéo de démo PERSISTANTE
# (livrable "20-30s montrant une performance réussie")
# ------------------------------------------------------------------
def record_demo_video(num_episodes: int, reward_threshold: float, max_attempts: int, fps: int, progress=gr.Progress()):
    """
    Appelle POST /record_demo. Toute la logique (rejouer jusqu'à trouver
    des parties réussies, écrire le .mp4 dans recordings/) est côté API :
    ce callback ne fait qu'appeler l'endpoint et afficher le résultat.
    """
    progress(0, desc="Génération en cours (peut prendre un moment selon le nombre de tentatives)...")
    resp = requests.post(
        f"{API_BASE_URL}/record_demo",
        json={
            "num_episodes": int(num_episodes),
            "reward_threshold": float(reward_threshold),
            "max_attempts": int(max_attempts),
            "fps": int(fps),
            "deterministic": True,
        },
        timeout=600,  # peut nécessiter plusieurs tentatives, donc plus long que /simulate
    )

    if resp.status_code == 422:
        # Aucune partie n'a atteint le seuil de réussite
        detail = resp.json().get("detail", "Échec de la génération.")
        return None, f"⚠️ {detail}"

    resp.raise_for_status()
    data = resp.json()

    warning = "" if data["target_reached"] else (
        f"\n\n⚠️ Seulement {data['episodes_used']}/{num_episodes} parties réussies trouvées "
        f"en {data['attempts']} tentatives (max_attempts atteint)."
    )

    summary = (
        f"**Vidéo enregistrée :** `{data['output_path']}`  \n"
        f"**Durée obtenue :** {data['duration_seconds']:.1f}s  \n"
        f"**Parties enchaînées :** {data['episodes_used']} "
        f"(récompenses : {', '.join(f'{r:.0f}' for r in data['episode_rewards'])})  \n"
        f"**Tentatives totales :** {data['attempts']}"
        f"{warning}"
    )
    return data["output_path"], summary


# ------------------------------------------------------------------
# Callbacks Dashboard : métriques de performance
# ------------------------------------------------------------------
def refresh_dashboard():
    summary = requests.get(f"{API_BASE_URL}/metrics/summary", timeout=30).json()
    history = requests.get(f"{API_BASE_URL}/metrics/history", timeout=30).json()
    decisions = requests.get(f"{API_BASE_URL}/metrics/decisions", timeout=30).json()

    # ---- Résumé texte ----
    if summary["n_episodes"] == 0:
        summary_md = "Aucune partie jouée pour le moment. Cliquez sur *Jouer une partie* dans l'onglet GUI."
    else:
        summary_md = (
            f"**Parties jouées :** {summary['n_episodes']}  \n"
            f"**Récompense moyenne :** {summary['mean_reward']:.2f}  \n"
            f"**Écart-type :** {summary['std_reward']:.2f}  \n"
            f"**Meilleur score :** {summary['best']:.2f}  \n"
            f"**Pire score :** {summary['worst']:.2f}"
        )

    # ---- Courbe de récompense par épisode ----
    if history["episodes"]:
        reward_df = pd.DataFrame({"Épisode": history["episodes"], "Récompense": history["rewards"]})
    else:
        reward_df = pd.DataFrame({"Épisode": [], "Récompense": []})

    # ---- Répartition des décisions par phase de vol (dernière partie) ----
    rows = []
    for phase, action_counts in decisions.get("phases", {}).items():
        for action, count in action_counts.items():
            rows.append({"Phase": phase, "Action": action, "Nombre": count})
    decisions_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Phase", "Action", "Nombre"])

    return summary_md, reward_df, decisions_df


# ------------------------------------------------------------------
# Construction de l'interface Gradio
# ------------------------------------------------------------------
with gr.Blocks(title="RL Agent — LunarLander") as demo:
    gr.Markdown("# 🚀 Tableau de bord de l'agent RL — LunarLander")
    gr.Markdown(
        "Le GUI et le dashboard ci-dessous ne font que dialoguer avec l'API "
        "(`/play`, `/simulate`, `/metrics/*`) : toute la logique RL est exécutée côté backend."
    )

    with gr.Tab("🎮 Voir une partie jouée"):
        with gr.Row():
            max_steps_input = gr.Slider(50, 1000, value=500, step=10, label="Nombre de pas maximum")
            frame_stride_input = gr.Slider(1, 10, value=3, step=1, label="1 frame gardée toutes les N (RAM)")
        play_btn = gr.Button("▶️ Jouer une partie", variant="primary")
        episode_summary = gr.Markdown()
        episode_video = gr.Video(label="Animation de la partie")

        play_btn.click(
            fn=play_episode_as_video,
            inputs=[max_steps_input, frame_stride_input],
            outputs=[episode_video, episode_summary],
        )

    with gr.Tab("🎬 Vidéo de démo (livrable)"):
        gr.Markdown(
            "Enchaîne plusieurs parties **réussies** (récompense ≥ seuil) pour "
            "constituer une vidéo continue de la durée voulue, et l'enregistre "
            "dans le dossier `recordings/` du projet (fichier permanent, "
            "contrairement à l'onglet précédent qui utilise un dossier temporaire)."
        )
        with gr.Row():
            num_episodes_input = gr.Slider(
                1, 10, value=3, step=1,
                label="Nombre de parties réussies à enchaîner (ajustez pour viser 20-30s)"
            )
            reward_threshold_input = gr.Slider(
                -100, 300, value=200, step=10,
                label="Seuil de récompense pour qu'une partie soit considérée « réussie »"
            )
        with gr.Row():
            max_attempts_input = gr.Slider(1, 100, value=20, step=1, label="Nombre max de tentatives")
            fps_input = gr.Slider(5, 30, value=15, step=1, label="Images par seconde de la vidéo")
        record_btn = gr.Button("🎥 Générer la vidéo de démo", variant="primary")
        record_summary = gr.Markdown()
        record_video = gr.Video(label="Vidéo de démo générée")

        record_btn.click(
            fn=record_demo_video,
            inputs=[num_episodes_input, reward_threshold_input, max_attempts_input, fps_input],
            outputs=[record_video, record_summary],
        )

    with gr.Tab("📊 Tableau de bord des performances"):
        refresh_btn = gr.Button("🔄 Rafraîchir")
        summary_md_out = gr.Markdown()
        reward_plot = gr.LinePlot(
            x="Épisode", y="Récompense", title="Récompense par épisode", height=300
        )
        decisions_plot = gr.BarPlot(
            x="Phase", y="Nombre", color="Action",
            title="Actions prises par phase de vol (dernière partie)", height=300
        )

        refresh_btn.click(
            fn=refresh_dashboard,
            outputs=[summary_md_out, reward_plot, decisions_plot],
        )
        demo.load(
            fn=refresh_dashboard,
            outputs=[summary_md_out, reward_plot, decisions_plot],
        )

    with gr.Tab("🔌 API"):
        gr.Markdown(
            "L'API FastAPI est servie sur ce même Space, sous les routes suivantes "
            "(documentation interactive : [/docs](/docs)) :\n\n"
            "- `POST /play` : `{\"state\": [...]}` -> `{\"action\": ...}`\n"
            "- `POST /simulate` : joue un épisode complet côté serveur\n"
            "- `POST /record_demo` : enchaîne des parties réussies et écrit une vidéo persistante (`recordings/`)\n"
            "- `GET /metrics/summary`, `/metrics/history`, `/metrics/decisions`\n"
        )

# On monte Gradio SUR l'application FastAPI existante (celle qui contient
# la logique RL), plutôt que de lancer deux serveurs séparés : une seule
# app sert à la fois l'API (/play, /simulate, /metrics/*) et le GUI (/).
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# En sdk: gradio, Hugging Face exécute ce script directement
# (équivalent de `python app.py`), donc __name__ == "__main__" est vrai
# ici : c'est CE bloc qui démarre le serveur, il n'y a rien d'autre qui
# écoute sur ce port en parallèle.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
