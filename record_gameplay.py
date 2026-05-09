import argparse
import os
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import gymnasium
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env_id = "ALE/SpaceInvaders-v5"
from vlm_agent import DEFAULT_MODEL, OpenAI, load_prompt, play_episode

ENV_ID = "ALE/SpaceInvaders-v5"


def _require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f'Missing required file: {path}')


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _append_frame(writer, env) -> None:
    frame = env.render()
    if frame is None:
        raise RuntimeError('Environment render() returned None. Check render_mode="rgb_array" support.')
    writer.append_data(frame)


def record_rl_agent(
    model_class,
    model_path: str,
    output_path: str,
    *,
    max_decisions: int = 10_000,
    seed: int = 42,
    fps: int = 30,
    sticky_action_repeat: int = 4,
) -> None:
    """
    Record one full gameplay episode for an SB3 Atari model.

    The model still receives the usual SB3-preprocessed frame-stacked observation.
    For video capture, we run a second raw RGB env with the same seed and replay the
    chosen action for 4 emulator steps to better match Atari preprocessing / frame skip.
    """
    _require_file(model_path)
    _ensure_parent_dir(output_path)

    vec_env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = model_class.load(model_path, env=vec_env)

    render_env = gymnasium.make(ENV_ID, render_mode='rgb_array')

    obs = vec_env.reset()
    render_env.reset(seed=seed)

    total_reward = 0.0
    written_frames = 0

    with imageio.get_writer(output_path, fps=fps) as writer:
        _append_frame(writer, render_env)
        written_frames += 1

        for step_idx in range(max_decisions):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action[0])

            obs, reward, done, _ = vec_env.step(action)
            total_reward += float(reward[0])

            render_done = False
            for _ in range(sticky_action_repeat):
                _, _, terminated, truncated, _ = render_env.step(action_int)
                _append_frame(writer, render_env)
                written_frames += 1
                render_done = terminated or truncated
                if render_done:
                    break

            if done[0] or render_done:
                print(f'Finished after {step_idx + 1} decisions.')
                break
        else:
            print(f'Reached max_decisions={max_decisions} before episode end.')

    vec_env.close()
    render_env.close()

    if written_frames <= 1:
        raise RuntimeError(f'Video {output_path} was not recorded correctly; only {written_frames} frame written.')

    print(
        f'Saved {output_path}: {written_frames} frames, '
        f'decision_reward={total_reward:.1f}, model={Path(model_path).name}'
    )


def record_vlm_agent(
    *,
    output_path: str = 'vlm_gameplay.mp4',
    max_steps: int = 2000,
    api_base: str = 'http://localhost:11434/v1',
    model: str = DEFAULT_MODEL,
    prompt_file: str = 'prompt.txt',
    fps: int = 30,
) -> None:
    _ensure_parent_dir(output_path)
    client = OpenAI(base_url=api_base, api_key=os.environ.get('OPENAI_API_KEY', 'EMPTY'))
    frames, reward = play_episode(
        client=client,
        model=model,
        system_prompt=load_prompt(prompt_file),
        max_steps=max_steps,
        record_frames=True,
    )
    if not frames:
        raise RuntimeError('VLM episode produced no frames.')

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f'Saved {output_path}: {len(frames)} frames, reward={reward:.1f}, model={model}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Record one gameplay video for each Assignment 4 agent.')
    parser.add_argument('--dqn-path', type=str, default='best_dqn.zip')
    parser.add_argument('--ppo-path', type=str, default='best_ppo.zip')
    parser.add_argument('--dqn-output', type=str, default='dqn_gameplay.mp4')
    parser.add_argument('--ppo-output', type=str, default='ppo_gameplay.mp4')
    parser.add_argument('--vlm-output', type=str, default='vlm_gameplay.mp4')
    parser.add_argument('--api-base', type=str, default='http://localhost:11434/v1')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--prompt-file', type=str, default='prompt.txt')
    parser.add_argument('--rl-max-decisions', type=int, default=10_000)
    parser.add_argument('--vlm-max-steps', type=int, default=2000)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--agents',
        nargs='+',
        choices=['dqn', 'ppo', 'vlm'],
        default=['dqn', 'ppo', 'vlm'],
        help='Which agents to record',
    )
    args = parser.parse_args()

    jobs: list[tuple[str, Callable[[], None]]] = []

    if 'dqn' in args.agents:
        jobs.append(
            (
                'DQN',
                lambda: record_rl_agent(
                    DQN,
                    args.dqn_path,
                    args.dqn_output,
                    max_decisions=args.rl_max_decisions,
                    seed=args.seed,
                    fps=args.fps,
                ),
            )
        )

    if 'ppo' in args.agents:
        jobs.append(
            (
                'PPO',
                lambda: record_rl_agent(
                    PPO,
                    args.ppo_path,
                    args.ppo_output,
                    max_decisions=args.rl_max_decisions,
                    seed=args.seed,
                    fps=args.fps,
                ),
            )
        )

    if 'vlm' in args.agents:
        jobs.append(
            (
                'VLM',
                lambda: record_vlm_agent(
                    output_path=args.vlm_output,
                    max_steps=args.vlm_max_steps,
                    api_base=args.api_base,
                    model=args.model,
                    prompt_file=args.prompt_file,
                    fps=args.fps,
                ),
            )
        )

    for name, job in jobs:
        print(f'=== Recording {name} ===')
        job()
        
if __name__ == '__main__':
    main()
