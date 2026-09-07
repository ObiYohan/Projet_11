# RL Agent — API + GUI + Dashboard (LunarLander-v3)

Projet local (démo) déployant un agent DQN entraîné sur `LunarLander-v3` avec :

1. **Une API FastAPI** — toute la logique RL (chargement du modèle, prédiction
   d'action, simulation d'épisode, calcul de métriques) y vit exclusivement.
2. **Un GUI Gradio** ("🎮 Voir une partie jouée") — affiche l'animation d'un
   épisode joué par l'agent, sous forme de vidéo.
3. **Un dashboard Gradio** ("📊 Tableau de bord") — courbe de récompense par
   épisode, moyenne/écart-type, et répartition des actions par phase de vol
   (haute altitude / approche / atterrissage).

Le GUI et le dashboard **n'exécutent aucune logique RL** : ils appellent
l'API en HTTP (`requests`), exactement comme le ferait n'importe quel client
externe, même si tout tourne dans un seul process sur votre machine.

## Architecture

```
project/
├── agent.py              # Logique RL : chargement modèle, predict_action(), run_episode()
├── metrics_store.py       # Persistance des métriques (CSV), lecture par chunks (RAM)
├── api.py                 # FastAPI : /play, /simulate, /record_demo, /metrics/*  <- LOGIQUE RL ICI
├── video_export.py        # Écriture des vidéos de démo persistantes (recordings/)
├── app.py                 # Gradio (GUI + dashboard), monté sur l'app FastAPI  <- AFFICHAGE SEULEMENT
├── train_and_export.py    # Script d'entraînement (voir aussi le notebook Colab)
├── model/
│   └── dqn_lunarlander.zip
├── data/
│   └── episodes.csv       # historique des parties jouées (créé automatiquement)
├── recordings/             # vidéos de démo persistantes (créé automatiquement)
└── requirements.txt
```

Un seul process sert à la fois l'API (`/play`, `/simulate`, `/metrics/*`,
`/docs`) et l'interface Gradio (`/`), grâce à
`gr.mount_gradio_app(fastapi_app, demo, path="/")`.

## Installation

```bash
# (recommandé) créez un environnement virtuel
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

`gymnasium[box2d]` (nécessaire pour LunarLander) demande parfois des
outils de compilation système. Si l'installation échoue sur `box2d-py` :

- **Windows** : installez les "Microsoft C++ Build Tools" avant `pip install`.
- **macOS** : `brew install swig` avant `pip install`.
- **Linux (Debian/Ubuntu)** : `sudo apt install swig build-essential` avant `pip install`.

## Utilisation

```bash
# 1. (Optionnel) Entraîner un modèle plus poussé que celui fourni
python train_and_export.py --timesteps 500000

# 2. Lancer le serveur (API + GUI + dashboard sur le même port)
python app.py
```

Puis ouvrez **http://localhost:7860** dans votre navigateur pour l'interface,
ou **http://localhost:7860/docs** pour la documentation Swagger interactive
de l'API (utile pour tester `/play` et `/simulate` directement, sans passer
par le GUI).

Pour changer de port :

```bash
PORT=8000 python app.py
```

## Endpoints de l'API

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Vérifie que le modèle est bien chargé |
| POST | `/play` | `{"state": [...]}` → `{"action": ...}` |
| POST | `/simulate` | Joue un épisode complet côté serveur, renvoie récompense, actions et frames (base64 JPEG) |
| POST | `/record_demo` | Enchaîne des parties **réussies** et écrit une vidéo `.mp4` persistante dans `recordings/` |
| GET | `/metrics/summary` | Moyenne, écart-type, meilleur/pire score sur toutes les parties jouées |
| GET | `/metrics/history` | Récompense épisode par épisode (pour la courbe) |
| GET | `/metrics/decisions` | Répartition des actions par phase de vol (dernière partie) |

Exemple d'appel direct (sans passer par le GUI) :

```bash
curl -X POST http://localhost:7860/play \
  -H "Content-Type: application/json" \
  -d '{"state": [0,0,0,0,0,0,0,0]}'
# -> {"action": 2}
```

## Générer la vidéo de démo (livrable "20-30s, performance réussie")

Un atterrissage réussi ne dure souvent que quelques secondes en vidéo —
pour couvrir 20-30s, `POST /record_demo` (onglet GUI "🎬 Vidéo de démo")
**enchaîne plusieurs parties réussies** dans un seul fichier, et l'écrit
dans le dossier **persistant** `recordings/` (contrairement aux vidéos de
l'onglet "🎮 Voir une partie jouée", qui vivent dans le dossier temporaire
du système et peuvent disparaître).

```bash
curl -X POST http://localhost:7860/record_demo \
  -H "Content-Type: application/json" \
  -d '{"num_episodes": 3, "reward_threshold": 200, "max_attempts": 20, "fps": 15}'
```

Paramètres à ajuster :
- **`num_episodes`** — nombre de parties réussies à enchaîner. Ajustez-le
  jusqu'à obtenir une `duration_seconds` dans la plage 20-30s (visible
  dans la réponse et dans l'onglet GUI).
- **`reward_threshold`** — récompense minimale pour qu'une partie compte
  comme "réussie" (200 = seuil "officiel" de LunarLander résolu). Si
  `/record_demo` renvoie une erreur 422 (aucune partie n'atteint ce seuil
  en `max_attempts` tentatives), baissez-le ou entraînez le modèle plus
  longtemps — le message d'erreur liste les récompenses obtenues pour vous
  aider à choisir.
- **`max_attempts`** — nombre max de parties jouées (réussies + ratées)
  avant d'abandonner, pour éviter une boucle infinie si le modèle est
  encore faible.

## Modèle utilisé : PPO (actions continues) par défaut

Le projet est configuré par défaut sur un **PPO** entraîné sur
`LunarLanderContinuous-v3` (action = 2 floats : poussée du moteur principal
et du moteur latéral), généralement plus performant que le DQN discret sur
cet environnement.

`agent.py` est agnostique à l'algorithme grâce à l'API commune
`model.predict()` de Stable-Baselines3. Pour revenir à un DQN discret
(`LunarLander-v3`) :

```bash
export RL_ENV_NAME=LunarLander-v3
export RL_MODEL_PATH=model/dqn_lunarlander.zip
export RL_MODEL_ALGO=DQN
python app.py
```

`/play` renvoie une liste de 2 floats avec PPO (continu) ou un entier avec
DQN (discret) — le GUI et le dashboard n'ont besoin d'aucune modification,
ils affichent ce que l'API renvoie.

⚠️ **Point de vigilance sur le dashboard "décisions par phase de vol"** :
compter des actions continues par valeur exacte n'aurait aucun sens (deux
floats sont presque toujours différents d'un pas à l'autre). `agent.py`
(`_describe_action`) regroupe donc chaque action en catégories lisibles
(`principal:fort/faible/off`, `lateral:gauche/centre/droite`) avant de les
compter — c'est ce regroupement, pas l'action brute, qui alimente
`/metrics/decisions` et l'historique CSV.

Entraînez votre propre modèle avec `train_and_export.py --algo PPO` (ou
`--algo DQN`) — voir `python train_and_export.py --help`.

## Points de vigilance RAM (gestion de la mémoire)

- **Frames** : `frame_stride` (paramètre de `/simulate`, réglable depuis le GUI)
  ne garde qu'une frame sur N plutôt que toutes les frames — réduit la RAM et
  la taille des réponses HTTP sans casser la fluidité perçue de l'animation.
- **Encodage vidéo** (`app.py`, `play_episode_as_video`) : les frames sont
  écrites une par une dans le fichier `.mp4` via `imageio.get_writer` (écriture
  en flux), jamais toutes chargées simultanément dans un unique tableau numpy.
- **Métriques** (`metrics_store.py`) : `load_history_chunks()` lit le CSV par
  blocs de 500 lignes (`pandas chunksize`) plutôt que d'un bloc, et
  `summary_stats()` calcule moyenne/écart-type de façon incrémentale (sans
  jamais charger tout l'historique en mémoire).

## Notes sur les modèles fournis

`model/ppo_lunarlander_continuous.zip` (utilisé par défaut) et
`model/dqn_lunarlander.zip` (alternative discrète) sont des modèles
d'entraînement **très courts** (~1000-3000 pas), suffisants pour valider
que toute la chaîne (API → GUI → dashboard) fonctionne, mais **pas encore
performants**. Remplacez-les par un modèle entraîné plus longtemps
(`python train_and_export.py --algo PPO --timesteps 500000`) avant toute
démonstration.
