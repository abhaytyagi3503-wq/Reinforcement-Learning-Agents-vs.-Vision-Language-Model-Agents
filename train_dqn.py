import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
import argparse
import os
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

ENV_ID = 'SpaceInvadersNoFrameskip-v4'


def make_env(n_envs: int, seed: int):
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


def main():
    parser = argparse.ArgumentParser(description='Train a stronger DQN for Space Invaders.')
    parser.add_argument('--timesteps', type=int, default=10_000_000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--eval-freq', type=int, default=250_000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=str, default='runs/dqn')
    parser.add_argument('--save-path', type=str, default='best_dqn.zip')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.n_envs, args.seed)
    eval_env = make_env(1, args.seed + 10_000)

    # Based on RL Zoo / Atari-style defaults, but made more submission-friendly:
    # - 10M default timesteps instead of 2M
    # - more frequent best-checkpoint selection against a dedicated eval env
    # - large replay warmup to stabilize learning before updates
    model = DQN(
        policy='CnnPolicy',
        env=train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=100_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.10,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        max_grad_norm=10.0,
        optimize_memory_usage=False,
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
        name_prefix='dqn_checkpoint',
        save_replay_buffer=True,
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
        best_model = DQN.load(str(best_model_path))
        best_model.save(target_stem)
    else:
        print('best_model.zip not found; saving final model instead.')
        model.save(target_stem)

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
