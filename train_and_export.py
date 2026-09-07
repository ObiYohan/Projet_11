"""
train_and_export.py
====================
Script à exécuter (ou coller dans votre notebook Colab) pour entraîner le
modèle et produire le .zip attendu par agent.py.

Deux modes, correspondant aux deux versions de LunarLander vues dans les
exercices précédents :
    --algo DQN --env LunarLander-v3              (action discrète)
    --algo PPO --env LunarLanderContinuous-v3     (action continue, par défaut)

Usage :
    python train_and_export.py --algo PPO --timesteps 500000
    python train_and_export.py --algo DQN --env LunarLander-v3 --timesteps 500000
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy


def build_dqn(env):
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=6.3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=-1,
        target_update_interval=250,
        exploration_fraction=0.12,
        exploration_final_eps=0.1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./logs/",
    )


def build_ppo(env):
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log="./logs/",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["DQN", "PPO"], default="PPO")
    parser.add_argument("--env", default=None, help="Par défaut : LunarLanderContinuous-v3 (PPO) ou LunarLander-v3 (DQN)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--output", default=None, help="Par défaut : model/ppo_lunarlander_continuous ou model/dqn_lunarlander")
    args = parser.parse_args()

    env_name = args.env or ("LunarLanderContinuous-v3" if args.algo == "PPO" else "LunarLander-v3")
    output = args.output or ("model/ppo_lunarlander_continuous" if args.algo == "PPO" else "model/dqn_lunarlander")

    env = gym.make(env_name)
    model = build_ppo(env) if args.algo == "PPO" else build_dqn(env)

    model.learn(total_timesteps=args.timesteps)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"\n📊 Récompense moyenne sur 100 épisodes : {mean_reward:.2f} +/- {std_reward:.2f}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    model.save(output)
    print(f"\n💾 Modèle sauvegardé sous {output}.zip")
    print(
        f"-> Pointez agent.py dessus via les variables d'environnement :\n"
        f"   RL_ENV_NAME={env_name}\n"
        f"   RL_MODEL_PATH={output}.zip\n"
        f"   RL_MODEL_ALGO={args.algo}"
    )

    env.close()


if __name__ == "__main__":
    main()
