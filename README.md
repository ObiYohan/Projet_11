# RL Agent — API + GUI + Dashboard (LunarLander-v3)

Projet local (démo) déployant un agent DQN entraîné sur `LunarLander-v3` avec :

1. **Une API FastAPI** — toute la logique RL (chargement du modèle, prédiction
   d'action, simulation d'épisode, calcul de métriques) y vit exclusivement.
2. **Un GUI Gradio** ("🎮 Voir une partie jouée") — affiche l'animation d'un
   épisode joué par l'agent (vidéo) et un **tableau des décisions pas-à-pas**
   (phase de vol, action, récompense, altitude à chaque instant de l'épisode).
3. **Un dashboard Gradio** ("📊 Tableau de bord") — graphique interactif
   d'un ou plusieurs signaux (récompense, observation, action) au fil du
   temps sur la dernière partie jouée, avec option de normalisation.
4. **Des paramètres d'environnement modifiables depuis l'interface**
   (gravité, vent, turbulence) — accordéon "⚙️ Paramètres de l'environnement",
   partagé entre le GUI et la génération de vidéo de démo.
5. **Une surimpression de métriques sur la vidéo** — affiche en direct sur
   l'image les signaux choisis (récompense, observation, action), pour
   l'aperçu temporaire comme pour la vidéo de démo persistante.

Le GUI et le dashboard **n'exécutent aucune logique RL** : ils appellent
l'API en HTTP (`requests`), exactement comme le ferait n'importe quel client
externe, même si tout tourne dans un seul process sur votre machine.

## Architecture

```
project/
├── agent.py              # Logique RL : chargement modèle, predict_action(), run_episode()
├── metrics_store.py       # Persistance des métriques (CSV), lecture par chunks (RAM)
├── signals.py              # Source unique des signaux disponibles (dashboard + surimpression)
├── overlay.py               # Dessine les métriques choisies sur une frame (utilitaire partagé)
├── api.py                 # FastAPI : /play, /simulate, /record_demo, /metrics/*  <- LOGIQUE RL ICI
├── video_export.py        # Écriture des vidéos de démo persistantes (recordings/), avec surimpression
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
| GET | `/metrics/timeline` | Historique pas-à-pas complet (observation + action + récompense) de la dernière partie |

Exemple d'appel direct (sans passer par le GUI) :

```bash
curl -X POST http://localhost:7860/play \
  -H "Content-Type: application/json" \
  -d '{"state": [0,0,0,0,0,0,0,0]}'
# -> {"action": 2}
```

## Modifier la physique de l'environnement

Depuis l'accordéon "⚙️ Paramètres de l'environnement" du GUI (ou directement
via l'API, champ `env_config` de `/simulate` et `/record_demo`) :

| Paramètre | Défaut | Bornes | Effet |
|---|---|---|---|
| `gravity` | -10.0 | strictement entre -12 et 0 (imposé par gymnasium) | Gravité lunaire ; plus proche de 0 = chute plus lente |
| `enable_wind` | `false` | — | Active des rafales de vent aléatoires |
| `wind_power` | 15.0 | 0-20 (recommandé) | Intensité du vent |
| `turbulence_power` | 1.5 | 0-2 (recommandé) | Intensité de la turbulence (rotation parasite) |

Une valeur de `gravity` hors bornes renvoie une erreur 422 explicite avant
même de créer l'environnement (validation Pydantic dans `api.py`).

```bash
curl -X POST http://localhost:7860/simulate \
  -H "Content-Type: application/json" \
  -d '{"max_steps": 500, "env_config": {"gravity": -6.0, "enable_wind": true, "wind_power": 18.0, "turbulence_power": 1.8}}'
```

## Tableau des décisions dans le temps

L'onglet "🎮 Voir une partie jouée" affiche, sous la vidéo, un tableau avec
une ligne par pas de l'épisode : phase de vol, action prise (regroupée en
catégories lisibles pour PPO continu — voir `agent.py::_describe_action`),
récompense instantanée et altitude. Ces données viennent du champ
`step_records` renvoyé par `/simulate` — aucun appel API supplémentaire
n'est nécessaire, le tableau est construit à partir de la même réponse que
la vidéo.

`step_records` contient en réalité l'**observation complète** à chaque pas
(8 dimensions de LunarLander, nommées : `obs_x`, `obs_y`, `obs_vx`, `obs_vy`,
`obs_angle`, `obs_angular_velocity`, `obs_leg1_contact`, `obs_leg2_contact`,
plus le vecteur brut sous `observation`) et les composantes d'action
séparées (`action_main`/`action_lateral` pour PPO continu, `action_discrete`
pour DQN) — pas seulement les champs affichés dans le tableau.

## Graphique "actions vs observations dans le temps"

L'onglet "📊 Tableau de bord" trace **un ou plusieurs signaux** au choix
(récompense, une dimension de l'observation, ou une composante de l'action)
en fonction du pas de l'épisode, pour la dernière partie jouée
(menu déroulant multi-sélection). Le menu ne propose que les signaux
réellement renseignés par le modèle chargé (les colonnes d'action continue
sont masquées avec un DQN discret, et inversement) — recalculé à chaque
clic sur "🔄 Rafraîchir" via `GET /metrics/timeline`. Changer la sélection
ne refait PAS d'appel API : les données de la dernière partie sont mises en
cache (`gr.State`) côté GUI et le graphique est reconstruit localement.

La case "Normaliser" centre-réduit chaque signal (z-score) avant de le
tracer — utile pour comparer sur le même graphique des signaux d'échelles
très différentes (ex : récompense ~[-100, 100] vs contact jambe ~[0, 1]),
sans que l'un écrase visuellement l'autre.

But : visualiser la corrélation entre ce que le modèle observe (ex :
l'angle du vaisseau, l'altitude) et l'action qu'il choisit en réaction —
utile pour repérer, par exemple, si l'agent augmente bien la poussée du
moteur principal quand l'altitude chute rapidement.

## Surimpression de métriques sur la vidéo

Les deux onglets vidéo ("🎮 Voir une partie jouée" et "🎬 Vidéo de démo")
ont un menu multi-sélection "Métriques en surimpression sur la vidéo" (les
mêmes signaux que le dashboard). Une fois sélectionnés, un petit panneau
apparaît en haut à gauche de chaque frame avec la valeur de chaque signal
à cet instant précis de l'épisode.

Le rendu (`overlay.py::draw_metrics_overlay`) est un utilitaire PARTAGÉ,
mais appelé à deux endroits différents selon la vidéo :
- **Aperçu temporaire** (`app.py`) : dessiné côté GUI, à partir des
  `step_records` déjà reçus dans la réponse de `/simulate` — pas d'appel
  API supplémentaire.
- **Vidéo de démo persistante** (`video_export.py`) : dessiné côté API,
  car cette vidéo est écrite entièrement côté serveur et ne transite
  jamais par le frontend. Le champ `overlay_signals` de `POST /record_demo`
  transporte la sélection.

Dans les deux cas, l'alignement entre une frame (potentiellement
sous-échantillonnée via `frame_stride`) et sa ligne de `step_records` se
fait via `frame_step_indices` (renvoyé par `agent.run_episode`), pour
éviter de recalculer une correspondance fragile à plusieurs endroits.

```bash
curl -X POST http://localhost:7860/record_demo \
  -H "Content-Type: application/json" \
  -d '{"num_episodes": 2, "reward_threshold": 200, "overlay_signals": ["Récompense", "Altitude (Y)"]}'
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

⚠️ **Point de vigilance sur le tableau "décisions dans le temps"** :
la colonne `action` (utilisée dans le tableau, et pour l'historique CSV des
métriques) regroupe les actions continues en catégories lisibles
(`principal:fort/faible/off`, `lateral:gauche/centre/droite`) via
`agent.py::_describe_action`, car compter des floats par valeur exacte
n'aurait aucun sens (deux floats sont presque toujours différents d'un pas
à l'autre). Le graphique "actions vs observations dans le temps" du
dashboard, lui, trace les valeurs numériques brutes (`action_main`,
`action_lateral`) — c'est ce qu'il faut pour voir une vraie courbe plutôt
qu'un nuage de catégories.

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
