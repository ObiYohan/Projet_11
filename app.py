"""
app.py
======
Point d'entrée du Hugging Face Space.

IMPORTANT (point de vigilance des consignes) : ce fichier ne contient
AUCUNE logique RL. Il ne fait que :
  1. monter l'application FastAPI (définie dans api.py) qui EXPOSE la logique,
  2. construire une interface Gradio dont les callbacks appellent l'API
     en HTTP (via `requests`), exactement comme le ferait n'importe quel
     client externe.

Cette séparation garantit que toute la logique RL reste testable et
réutilisable indépendamment du frontend (on pourrait brancher un tout
autre GUI sur la même API sans rien changer côté modèle).

Exception assumée : la surimpression de métriques sur la vidéo d'aperçu
("Voir une partie jouée") est dessinée ICI, via overlay.py. Ce n'est pas
de la logique RL — c'est du rendu d'image pur à partir de données déjà
calculées et déjà reçues de l'API (step_records) — au même titre que
l'encodage de la vidéo elle-même, qui se fait déjà dans ce fichier. Pour
la vidéo de démo PERSISTANTE (/record_demo), la surimpression est en
revanche dessinée côté API (video_export.py), car cette vidéo est écrite
entièrement côté serveur sans jamais transiter par le frontend.

Architecture du Space (un seul process, un seul port) :
    Ce Space est en sdk: gradio (pas Docker) : Hugging Face exécute
    directement `python app.py`. C'est pour ça que ce fichier se termine
    par un bloc `if __name__ == "__main__": uvicorn.run(...)` — c'est lui
    qui démarre effectivement le serveur, sur le port attendu par HF Spaces
    (variable d'environnement PORT, 7860 par défaut).

    uvicorn sert `app` (FastAPI, avec Gradio monté dessus) sur ce port
    -> les routes /play, /simulate, /metrics/* sont la logique RL (api.py)
    -> la route "/" sert l'interface Gradio (montée ci-dessous), qui elle
       même appelle http://127.0.0.1:{PORT}/... pour tout ce qu'elle affiche
"""

import base64
import io
import os
import tempfile
import time

import gradio as gr
import numpy as np
import pandas as pd
import requests
import imageio.v2 as imageio
from PIL import Image

from api import app as fastapi_app  # la logique RL vit ici, pas dans ce fichier
from overlay import draw_metrics_overlay
from signals import SIGNAL_COLUMNS

# En interne, le GUI appelle l'API sur le même process (boucle locale).
# PORT doit rester cohérent avec le port passé à uvicorn.run() en bas de ce fichier.
PORT = int(os.environ.get("PORT", 7860))
API_BASE_URL = os.environ.get("RL_API_BASE_URL", f"http://127.0.0.1:{PORT}")


# ------------------------------------------------------------------
# Callbacks GUI : "voir une partie jouée"
# ------------------------------------------------------------------
def _env_config_payload(gravity: float, enable_wind: bool, wind_power: float, turbulence_power: float) -> dict:
    """Construit le payload env_config envoyé à l'API, partagé par tous les callbacks."""
    return {
        "gravity": float(gravity),
        "enable_wind": bool(enable_wind),
        "wind_power": float(wind_power),
        "turbulence_power": float(turbulence_power),
    }


def play_episode_as_video(
    max_steps: int,
    frame_stride: int,
    overlay_labels: list,
    gravity: float,
    enable_wind: bool,
    wind_power: float,
    turbulence_power: float,
):
    """
    Construit un court fichier vidéo (mp4) à partir des frames reçues de
    l'API, avec en option une surimpression des signaux choisis (dessinée
    ici, voir la note en tête de fichier). Utilise imageio en générateur
    d'écriture (pas de gros tableau numpy unique en mémoire) pour rester
    léger en RAM.

    Renvoie aussi un tableau des décisions pas-à-pas (step_records reçus
    de l'API), pour visualiser les actions prises au fil de l'épisode.
    """
    resp = requests.post(
        f"{API_BASE_URL}/simulate",
        json={
            "deterministic": True,
            "max_steps": int(max_steps),
            "frame_stride": int(frame_stride),
            "include_frames": True,
            "env_config": _env_config_payload(gravity, enable_wind, wind_power, turbulence_power),
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    frame_step_indices = data.get("frame_step_indices", [])
    step_records = data.get("step_records", [])

    video_path = os.path.join(tempfile.gettempdir(), f"episode_{data['episode_id']}_{int(time.time())}.mp4")
    with imageio.get_writer(video_path, fps=15) as writer:
        for i, b64 in enumerate(data["frames_base64"]):
            img = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))

            if overlay_labels:
                step_idx = frame_step_indices[i] if i < len(frame_step_indices) else None
                record = step_records[step_idx] if step_idx is not None and step_idx < len(step_records) else None
                if record is not None:
                    img = draw_metrics_overlay(img, record, overlay_labels)

            writer.append_data(img)  # écrite frame par frame, pas de buffer global

    summary = (
        f"**Épisode #{data['episode_id']}** — "
        f"Récompense totale : **{data['total_reward']:.2f}** — "
        f"Durée : {data['steps']} pas"
    )

    # Tableau des décisions dans le temps : un pas = une ligne
    decisions_df = pd.DataFrame(step_records)
    if not decisions_df.empty:
        decisions_df = decisions_df[["step", "phase", "action", "reward", "altitude"]].rename(
            columns={
                "step": "Pas",
                "phase": "Phase",
                "action": "Action",
                "reward": "Récompense",
                "altitude": "Altitude",
            }
        )

    return video_path, summary, decisions_df


# ------------------------------------------------------------------
# Callback GUI : enregistrement de la vidéo de démo PERSISTANTE
# (livrable "20-30s montrant une performance réussie")
# ------------------------------------------------------------------
def record_demo_video(
    num_episodes: int,
    reward_threshold: float,
    max_attempts: int,
    fps: int,
    overlay_labels: list,
    gravity: float,
    enable_wind: bool,
    wind_power: float,
    turbulence_power: float,
    progress=gr.Progress(),
):
    """
    Appelle POST /record_demo. Toute la logique (rejouer jusqu'à trouver
    des parties réussies, écrire le .mp4 dans recordings/, y compris la
    surimpression des métriques) est côté API : ce callback ne fait
    qu'appeler l'endpoint et afficher le résultat.
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
            "env_config": _env_config_payload(gravity, enable_wind, wind_power, turbulence_power),
            "overlay_signals": overlay_labels or [],
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
# Callbacks Dashboard : actions vs observations dans le temps
# ------------------------------------------------------------------
def _build_chart_df(raw_df: pd.DataFrame, selected_signals: list, normalize: bool) -> pd.DataFrame:
    """
    Construit le DataFrame long format (Pas, Valeur, Signal) attendu par le
    LinePlot pour un ou plusieurs signaux choisis — une couleur par signal
    (paramètre `color="Signal"` du LinePlot).

    normalize=True centre-réduit chaque signal indépendamment (z-score),
    utile pour comparer des signaux d'échelles très différentes sur le même
    graphique (ex : récompense ~[-100, 100] vs contact jambe ~[0, 1]).
    """
    if raw_df.empty or not selected_signals:
        return pd.DataFrame({"Pas": [], "Valeur": [], "Signal": []})

    frames = []
    for label in selected_signals:
        col = SIGNAL_COLUMNS.get(label)
        if col is None or col not in raw_df.columns:
            continue
        values = raw_df[col].astype(float)
        if normalize:
            std = values.std()
            values = (values - values.mean()) / std if std > 1e-9 else values * 0.0
        frames.append(pd.DataFrame({"Pas": raw_df["step"], "Valeur": values, "Signal": label}))

    if not frames:
        return pd.DataFrame({"Pas": [], "Valeur": [], "Signal": []})
    return pd.concat(frames, ignore_index=True)


def refresh_dashboard(selected_signals: list, normalize: bool):
    """
    Récupère /metrics/summary (stats agrégées) et /metrics/timeline (données
    pas-à-pas brutes de la dernière partie), puis construit le graphique
    pour les signaux actuellement sélectionnés. La liste des signaux
    proposés est recalculée à chaque rafraîchissement, pour ne montrer que
    les colonnes réellement renseignées par le modèle chargé (discret vs continu).
    """
    summary = requests.get(f"{API_BASE_URL}/metrics/summary", timeout=30).json()
    timeline = requests.get(f"{API_BASE_URL}/metrics/timeline", timeout=30).json()

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

    records = timeline.get("step_records", [])
    raw_df = pd.DataFrame(records) if records else pd.DataFrame()

    valid_options = [
        label for label, col in SIGNAL_COLUMNS.items()
        if col in raw_df.columns and raw_df[col].notna().any()
    ] or ["Récompense"]

    selected_signals = [s for s in (selected_signals or []) if s in valid_options] or [valid_options[0]]

    chart_df = _build_chart_df(raw_df, selected_signals, normalize)

    return (
        summary_md,
        raw_df,
        gr.update(choices=valid_options, value=selected_signals),
        gr.update(value=chart_df),
    )


def update_timeline_chart(raw_df: pd.DataFrame, selected_signals: list, normalize: bool):
    """Change juste les signaux affichés, à partir des données DÉJÀ chargées (pas de nouvel appel API)."""
    chart_df = _build_chart_df(raw_df, selected_signals, normalize)
    return gr.update(value=chart_df)


# ------------------------------------------------------------------
# Construction de l'interface Gradio
# ------------------------------------------------------------------
with gr.Blocks(title="RL Agent — LunarLander") as demo:
    gr.Markdown("# 🚀 Tableau de bord de l'agent RL — LunarLander")
    gr.Markdown(
        "Le GUI et le dashboard ci-dessous ne font que dialoguer avec l'API "
        "(`/play`, `/simulate`, `/metrics/*`) : toute la logique RL est exécutée côté backend."
    )

    # Paramètres d'environnement PARTAGÉS entre les onglets "Voir une partie
    # jouée" et "Vidéo de démo" (mêmes composants Gradio référencés dans les
    # deux callbacks .click() plus bas).
    with gr.Accordion("⚙️ Paramètres de l'environnement (physique de LunarLander)", open=False):
        gr.Markdown(
            "Modifie la physique de l'environnement pour les prochaines parties jouées "
            "(GUI et vidéo de démo). N'affecte pas `/play` (qui ne fait qu'une prédiction "
            "sur un état déjà donné, sans environnement)."
        )
        with gr.Row():
            gravity_input = gr.Slider(-11.9, -0.1, value=-10.0, step=0.1, label="Gravité (doit rester entre -12 et 0)")
            enable_wind_input = gr.Checkbox(value=False, label="Activer le vent")
        with gr.Row():
            wind_power_input = gr.Slider(0.0, 20.0, value=15.0, step=0.5, label="Intensité du vent")
            turbulence_power_input = gr.Slider(0.0, 2.0, value=1.5, step=0.1, label="Intensité de la turbulence")

    env_config_inputs = [gravity_input, enable_wind_input, wind_power_input, turbulence_power_input]

    with gr.Tab("🎮 Voir une partie jouée"):
        with gr.Row():
            max_steps_input = gr.Slider(50, 1000, value=500, step=10, label="Nombre de pas maximum")
            frame_stride_input = gr.Slider(1, 10, value=3, step=1, label="1 frame gardée toutes les N (RAM)")
        overlay_input = gr.Dropdown(
            choices=list(SIGNAL_COLUMNS.keys()),
            value=[],
            multiselect=True,
            label="Métriques en surimpression sur la vidéo (optionnel)",
        )
        play_btn = gr.Button("▶️ Jouer une partie", variant="primary")
        episode_summary = gr.Markdown()
        episode_video = gr.Video(label="Animation de la partie")
        decisions_table = gr.Dataframe(
            label="Décisions dans le temps (une ligne = un pas de l'épisode)",
            wrap=True,
        )

        play_btn.click(
            fn=play_episode_as_video,
            inputs=[max_steps_input, frame_stride_input, overlay_input, *env_config_inputs],
            outputs=[episode_video, episode_summary, decisions_table],
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
        overlay_demo_input = gr.Dropdown(
            choices=list(SIGNAL_COLUMNS.keys()),
            value=[],
            multiselect=True,
            label="Métriques en surimpression sur la vidéo (optionnel)",
        )
        record_btn = gr.Button("🎥 Générer la vidéo de démo", variant="primary")
        record_summary = gr.Markdown()
        record_video = gr.Video(label="Vidéo de démo générée")

        record_btn.click(
            fn=record_demo_video,
            inputs=[
                num_episodes_input, reward_threshold_input, max_attempts_input, fps_input,
                overlay_demo_input, *env_config_inputs,
            ],
            outputs=[record_video, record_summary],
        )

    with gr.Tab("📊 Tableau de bord des performances"):
        gr.Markdown(
            "Visualise les réactions du modèle par rapport aux observations faites, "
            "au fil du temps sur la dernière partie jouée : choisissez un ou plusieurs "
            "signaux (récompense, une dimension de l'observation, une composante de "
            "l'action) pour voir leur évolution pas à pas."
        )
        refresh_btn = gr.Button("🔄 Rafraîchir")
        summary_md_out = gr.Markdown()
        timeline_state = gr.State(pd.DataFrame())  # cache local des step_records déjà récupérés
        with gr.Row():
            signal_dropdown = gr.Dropdown(
                choices=list(SIGNAL_COLUMNS.keys()),
                value=["Récompense"],
                multiselect=True,
                label="Signaux à visualiser dans le temps",
            )
            normalize_checkbox = gr.Checkbox(
                value=False,
                label="Normaliser",
            )
        timeline_plot = gr.LinePlot(
            x="Pas", y="Valeur", color="Signal", title="Signaux dans le temps (dernière partie)", height=350
        )

        refresh_btn.click(
            fn=refresh_dashboard,
            inputs=[signal_dropdown, normalize_checkbox],
            outputs=[summary_md_out, timeline_state, signal_dropdown, timeline_plot],
        )
        demo.load(
            fn=refresh_dashboard,
            inputs=[signal_dropdown, normalize_checkbox],
            outputs=[summary_md_out, timeline_state, signal_dropdown, timeline_plot],
        )
        signal_dropdown.change(
            fn=update_timeline_chart,
            inputs=[timeline_state, signal_dropdown, normalize_checkbox],
            outputs=[timeline_plot],
        )
        normalize_checkbox.change(
            fn=update_timeline_chart,
            inputs=[timeline_state, signal_dropdown, normalize_checkbox],
            outputs=[timeline_plot],
        )

    with gr.Tab("🔌 API"):
        gr.Markdown(
            "L'API FastAPI est servie sur ce même Space, sous les routes suivantes "
            "(documentation interactive : [/docs](/docs)) :\n\n"
            "- `POST /play` : `{\"state\": [...]}` -> `{\"action\": ...}`\n"
            "- `POST /simulate` : joue un épisode complet côté serveur (accepte `env_config` : gravity, enable_wind, wind_power, turbulence_power)\n"
            "- `POST /record_demo` : enchaîne des parties réussies et écrit une vidéo persistante (`recordings/`), avec `overlay_signals` optionnel\n"
            "- `GET /metrics/summary`, `/metrics/timeline`\n"
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
