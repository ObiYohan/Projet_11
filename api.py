"""
api.py
======
API FastAPI. C'est le SEUL endroit (avec agent.py / metrics_store.py) où
la logique RL s'exécute : chargement du modèle, prédiction d'action,
simulation d'épisode, calcul des métriques.

Endpoints :
- GET  /health              : vérifie que l'API et le modèle sont prêts
- POST /play                : reçoit un état, renvoie une action (requis par les consignes)
- POST /simulate             : joue un épisode complet côté serveur, renvoie
                                récompense, actions, et frames pour l'animation
- POST /record_demo          : enchaîne plusieurs épisodes RÉUSSIS et écrit
                                une vidéo .mp4 persistante (dossier recordings/)
                                — c'est l'endpoint pour le livrable "vidéo 20-30s"
- GET  /metrics/summary      : stats agrégées (moyenne, écart-type, meilleur/pire score)
- GET  /metrics/timeline     : historique pas-à-pas (observation + action + récompense) de la dernière partie
"""

import base64
import io
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

import metrics_store
import video_export
from agent import Agent

# ------------------------------------------------------------------
# Chargement du modèle : une seule fois, au démarrage de l'API
# (lifespan de FastAPI), jamais à chaque requête.
# ------------------------------------------------------------------
_agent_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _agent_holder["agent"] = Agent()
    except FileNotFoundError as e:
        # On laisse l'API démarrer quand même pour que /health explique le problème
        # plutôt que de crasher tout le Space au boot.
        _agent_holder["agent"] = None
        _agent_holder["load_error"] = str(e)
    yield
    _agent_holder.clear()


app = FastAPI(title="RL Agent API", lifespan=lifespan)


def get_agent() -> Agent:
    agent = _agent_holder.get("agent")
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=_agent_holder.get("load_error", "Modèle non chargé."),
        )
    return agent


# ------------------------------------------------------------------
# Schémas de requête / réponse
# ------------------------------------------------------------------
class EnvConfig(BaseModel):
    """
    Paramètres physiques de LunarLander, modifiables depuis l'interface.
    Les bornes reprennent celles imposées par gymnasium (gravity) ou
    recommandées (wind_power, turbulence_power) — une valeur hors bornes
    est rejetée avec une erreur 422 explicite avant même d'atteindre
    l'environnement.
    """
    gravity: float = Field(-10.0, gt=-12.0, lt=0.0, description="Gravité (doit être entre -12 et 0, exclus)")
    enable_wind: bool = Field(False, description="Active le vent (rafales aléatoires)")
    wind_power: float = Field(15.0, ge=0.0, le=20.0, description="Intensité du vent (recommandé : 0-20)")
    turbulence_power: float = Field(1.5, ge=0.0, le=2.0, description="Intensité de la turbulence (recommandé : 0-2)")


class PlayRequest(BaseModel):
    state: list[float] = Field(..., description="État brut de l'environnement (ex: 8 floats pour LunarLander)")
    deterministic: bool = True


class PlayResponse(BaseModel):
    action: object  # int (DQN) ou list[float] (PPO continu)


class SimulateRequest(BaseModel):
    deterministic: bool = True
    max_steps: int = 1000
    frame_stride: int = Field(3, ge=1, description="1 frame conservée toutes les N, pour limiter la RAM")
    include_frames: bool = True
    env_config: Optional[EnvConfig] = None


class SimulateResponse(BaseModel):
    episode_id: int
    total_reward: float
    steps: int
    actions: list
    frames_base64: Optional[list[str]] = None  # images JPEG encodées, prêtes pour affichage GUI
    frame_step_indices: list[int] = []  # frames_base64[i] <-> step_records[frame_step_indices[i]]
    step_records: list = []  # historique pas-à-pas (step, phase, action, reward, altitude) pour le tableau des décisions


class RecordDemoRequest(BaseModel):
    num_episodes: int = Field(3, ge=1, le=10, description="Nombre de parties RÉUSSIES à enchaîner dans la vidéo")
    reward_threshold: float = Field(200.0, description="Récompense totale minimale pour qu'une partie soit considérée 'réussie'")
    max_attempts: int = Field(20, ge=1, le=100, description="Nombre max de parties jouées avant d'abandonner")
    fps: int = Field(15, ge=5, le=60)
    deterministic: bool = True
    env_config: Optional[EnvConfig] = None
    overlay_signals: list[str] = Field(
        default_factory=list,
        description="Labels de signaux (clés de signals.SIGNAL_COLUMNS) à afficher en surimpression sur la vidéo. Vide = pas de surimpression.",
    )


class RecordDemoResponse(BaseModel):
    output_path: str
    duration_seconds: float
    episodes_used: int
    episode_rewards: list[float]         # récompenses des épisodes RETENUS dans la vidéo
    attempts: int                        # nombre total de parties jouées (réussies + ratées)
    all_attempt_rewards: list[float]     # récompenses de TOUTES les tentatives, pour diagnostic
    target_reached: bool                 # False si max_attempts atteint avant num_episodes réussis


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health")
def health():
    agent = _agent_holder.get("agent")
    return {"status": "ok" if agent is not None else "model_not_loaded", "env": getattr(agent, "env_name", None)}


@app.post("/play", response_model=PlayResponse)
def play(req: PlayRequest):
    """Endpoint demandé par les consignes : état -> action."""
    agent = get_agent()
    action = agent.predict_action(req.state, deterministic=req.deterministic)
    return PlayResponse(action=action)


def _frame_to_base64_jpeg(frame: np.ndarray, quality: int = 70) -> str:
    """Encode une frame RGB (numpy array) en JPEG base64, pour transport HTTP léger."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """
    Joue un épisode complet côté serveur et enregistre ses métriques.
    Toute la logique (jeu de l'épisode, capture des frames, calcul des
    métriques) reste ici : le GUI ne fait qu'appeler cet endpoint et
    afficher ce qu'il reçoit.
    """
    agent = get_agent()
    result = agent.run_episode(
        deterministic=req.deterministic,
        max_steps=req.max_steps,
        frame_stride=req.frame_stride,
        env_kwargs=req.env_config.dict() if req.env_config else None,
    )

    episode_id = metrics_store.append_episode(
        total_reward=result["total_reward"],
        steps=result["steps"],
        # on compte les LABELS d'action (déjà regroupés pour le cas continu),
        # pas les actions brutes : deux floats sont presque toujours
        # différents d'un pas à l'autre, un comptage par valeur exacte ne
        # produirait quasiment aucun regroupement utile pour un PPO continu.
        actions=[rec["action"] for rec in result["step_records"]],
    )

    frames_b64 = None
    if req.include_frames:
        # on encode les frames une par une (générateur implicite via la
        # boucle) plutôt que de garder toutes les images PIL en mémoire
        # simultanément avant l'encodage
        frames_b64 = [_frame_to_base64_jpeg(f) for f in result["frames"]]

    # on stocke aussi les step_records pour le endpoint /metrics/timeline
    _last_episode_holder["records"] = result["step_records"]

    return SimulateResponse(
        episode_id=episode_id,
        total_reward=result["total_reward"],
        steps=result["steps"],
        actions=result["actions"],
        frames_base64=frames_b64,
        frame_step_indices=result["frame_step_indices"],
        step_records=result["step_records"],
    )


_last_episode_holder: dict = {"records": []}


@app.post("/record_demo", response_model=RecordDemoResponse)
def record_demo(req: RecordDemoRequest):
    """
    Enchaîne des parties jusqu'à en avoir `num_episodes` de RÉUSSIES
    (récompense >= reward_threshold), et écrit une vidéo .mp4 unique dans
    le dossier PERSISTANT recordings/ (contrairement aux vidéos générées
    à la volée par le GUI, qui vivent dans le dossier temporaire du système).

    C'est l'endpoint à utiliser pour produire le livrable "vidéo de 20-30s
    montrant une performance réussie" : ajustez num_episodes pour atteindre
    la durée voulue (chaque partie réussie dure typiquement quelques
    secondes ; enchaînez-en plusieurs pour couvrir 20-30s).
    """
    agent = get_agent()

    collected = agent.collect_successful_episodes(
        num_episodes=req.num_episodes,
        reward_threshold=req.reward_threshold,
        max_attempts=req.max_attempts,
        deterministic=req.deterministic,
        env_kwargs=req.env_config.model_dump() if req.env_config else None,
    )

    if not collected["episodes"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Aucune partie réussie sur {collected['attempts']} tentatives "
                f"(seuil : {req.reward_threshold}). Récompenses obtenues : "
                f"{collected['all_attempt_rewards']}. Baissez reward_threshold "
                f"ou entraînez le modèle plus longtemps."
            ),
        )

    video_result = video_export.build_demo_video(
        collected["episodes"], fps=req.fps, overlay_signals=req.overlay_signals
    )

    return RecordDemoResponse(
        output_path=video_result["output_path"],
        duration_seconds=video_result["duration_seconds"],
        episodes_used=len(collected["episodes"]),
        episode_rewards=[ep["total_reward"] for ep in collected["episodes"]],
        attempts=collected["attempts"],
        all_attempt_rewards=collected["all_attempt_rewards"],
        target_reached=collected["target_reached"],
    )


@app.get("/metrics/summary")
def metrics_summary():
    return metrics_store.summary_stats()


@app.get("/metrics/history")
def metrics_history():
    """Historique épisode par épisode, pour tracer la courbe de récompense."""
    df = metrics_store.load_history()
    if df.empty:
        return {"episodes": [], "rewards": []}
    return {
        "episodes": df["episode_id"].tolist(),
        "rewards": df["total_reward"].tolist(),
        "steps": df["steps"].tolist(),
    }


@app.get("/metrics/timeline")
def metrics_timeline():
    """
    Historique pas-à-pas COMPLET (observation + action + récompense) de la
    dernière partie simulée (voir agent.run_episode -> step_records).
    Alimente le graphique "actions vs observations dans le temps" du GUI :
    contrairement à l'ancien /metrics/decisions (agrégation par phase), on
    renvoie ici les données brutes, pour que le GUI puisse tracer n'importe
    quel signal (récompense, une dimension de l'observation, une composante
    de l'action) en fonction du temps, au choix de l'utilisateur.
    """
    return {"step_records": _last_episode_holder.get("records", [])}
