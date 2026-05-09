import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import VecFrameStack

ENV_ID = 'SpaceInvadersNoFrameskip-v4'


def make_env(n_envs: int, seed: int):
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


def main():
    parser = argparse.ArgumentParser(description='Train a stronger PPO for Space Invaders.')
    parser.add_argument('--timesteps', type=int, default=10_000_000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--eval-freq', type=int, default=250_000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=str, default='runs/ppo')
    parser.add_argument('--save-path', type=str, default='best_ppo.zip')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.n_envs, args.seed)
    eval_env = make_env(1, args.seed + 20_000)

    # Starts from RL Zoo Atari PPO defaults and keeps the classic linear schedules
    # that usually help stabilize Atari PPO late in training.
    model = PPO(
        policy='CnnPolicy',
        env=train_env,
        learning_rate=get_linear_fn(2.5e-4, 2.5e-5, 1.0),
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=get_linear_fn(0.1, 0.01, 1.0),
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(normalize_images=False),
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
        device='auto',
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=max(500_000 // args.n_envs, 1),
        save_path=str(log_dir),
        name_prefix='ppo_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, ckpt_callback],
        progress_bar=True,
    )

    best_model_path = log_dir / 'best_model.zip'
    target_stem = args.save_path[:-4] if args.save_path.endswith('.zip') else args.save_path
    if best_model_path.exists():
        print(f'Using best eval checkpoint from {best_model_path}')
        best_model = PPO.load(str(best_model_path))
        best_model.save(target_stem)
    else:
        print('best_model.zip not found; saving final model instead.')
        model.save(target_stem)

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
