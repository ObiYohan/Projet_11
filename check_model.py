"""Petit script de diagnostic : vérifie quel algorithme correspond réellement
à un fichier .zip Stable-Baselines3, sans essayer de le charger (donc sans
planter même s'il ne correspond pas à l'algo attendu).

Usage :
    python check_model.py model/dqn_lunarlander.zip
"""
import sys
import zipfile
import json

path = sys.argv[1] if len(sys.argv) > 1 else "model/dqn_lunarlander.zip"

with zipfile.ZipFile(path) as z:
    with z.open("data") as f:
        data = json.load(f)

policy_serialized = data.get("policy_class", {})
module = policy_serialized.get("__module__", "?")
name = policy_serialized.get("__doc__", "")[:60].strip().replace("\n", " ")

print(f"Fichier         : {path}")
print(f"Module policy   : {module}")

if "dqn" in module:
    print("=> C'est un modèle DQN (compatible avec RL_MODEL_ALGO=DQN)")
elif "ppo" in module or "ActorCritic" in str(policy_serialized):
    print("=> C'est un modèle PPO (nécessite RL_MODEL_ALGO=PPO)")
else:
    print(f"=> Type non reconnu automatiquement : {module}")
