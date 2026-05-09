import argparse
import base64
import io
import json
import os
import re
from collections import Counter, deque
from typing import Deque, List, Tuple
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
import gymnasium
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw

ENV_ID = 'SpaceInvadersNoFrameskip-v4'
ACTIONS = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'RIGHT',
    3: 'LEFT',
    4: 'RIGHTFIRE',
    5: 'LEFTFIRE',
}
MOVEMENT_ACTIONS = {2, 3, 4, 5}
LEFT_ACTIONS = {3, 5}
RIGHT_ACTIONS = {2, 4}
FIRE_ACTIONS = {1, 4, 5}
DEFAULT_MODEL = os.environ.get('QWEN_MODEL', 'Qwen/Qwen3.5-2B')
DEFAULT_SYSTEM_PROMPT = """You are an ultra-concise Atari Space Invaders control policy.
You will receive one tiled image containing 4 chronological RGB frames, oldest to newest.
Return exactly one JSON object with an integer action from 0 to 5.
Priorities: survive, dodge bullets, stay under a useful lane, and fire often when safe.
Avoid useless left-right oscillation.
Prefer RIGHTFIRE or LEFTFIRE over pure movement when safe.
Format exactly: {\"action\": 4, \"reason\": \"short text\"}
"""


class VLMAtariWrapper:
    def __init__(self, env_id: str = ENV_ID, frame_skip: int = 4, stack_size: int = 4):
        self.env = gymnasium.make(env_id, render_mode='rgb_array')
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.frame_buffer = deque(maxlen=stack_size)

    def reset(self):
        _, info = self.env.reset()
        frame = self.env.render()
        for _ in range(self.stack_size):
            self.frame_buffer.append(frame)
        return list(self.frame_buffer), info

    def step(self, action: int):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            frame = self.env.render()
            self.frame_buffer.append(frame)
            if terminated or truncated:
                break
        return list(self.frame_buffer), total_reward, terminated, truncated, info

    def close(self):
        self.env.close()


from typing import Optional

def load_prompt(path: Optional[str]) -> str:
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return DEFAULT_SYSTEM_PROMPT


def make_tiled_image(frames: List[np.ndarray]) -> Image.Image:
    pil_frames = [Image.fromarray(frame) for frame in frames]
    width, height = pil_frames[0].size
    canvas = Image.new('RGB', (width * 2, height * 2), color=(0, 0, 0))
    positions = [(0, 0), (width, 0), (0, height), (width, height)]
    draw = ImageDraw.Draw(canvas)

    for idx, (img, pos) in enumerate(zip(pil_frames, positions), start=1):
        canvas.paste(img, pos)
        x0, y0 = pos
        draw.rectangle([x0 + 4, y0 + 4, x0 + 84, y0 + 30], fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 8), f'Frame {idx}', fill=(255, 255, 255))
    return canvas


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    payload = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{payload}'


def extract_action(text: str) -> int:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            action = int(obj['action'])
            if action in ACTIONS:
                return action
    except Exception:
        pass

    match = re.search(r'\b([0-5])\b', text)
    if match:
        return int(match.group(1))
    return 1


def summarize_history(recent_history: List[Tuple[int, float]]) -> str:
    if not recent_history:
        return 'No recent history.'
    lines = []
    for i, (action, reward) in enumerate(recent_history[-6:], start=1):
        lines.append(f'{i}. {ACTIONS[action]} ({action}), reward={reward:.1f}')
    return '\n'.join(lines)


def apply_action_guardrails(proposed_action: int, recent_actions: List[int]) -> int:
    if not recent_actions:
        return proposed_action

    last_action = recent_actions[-1]
    recent_window = recent_actions[-4:]

    # Break pure left-right ping-pong.
    if len(recent_actions) >= 2:
        prev_action = recent_actions[-2]
        if proposed_action in LEFT_ACTIONS and last_action in RIGHT_ACTIONS and prev_action in LEFT_ACTIONS:
            return 1
        if proposed_action in RIGHT_ACTIONS and last_action in LEFT_ACTIONS and prev_action in RIGHT_ACTIONS:
            return 1

    # If we have been idle for too long, bias toward firing.
    if len(recent_window) == 4 and all(a == 0 for a in recent_window) and proposed_action == 0:
        return 1

    # If we keep moving without firing, upgrade to move+fire when possible.
    if len(recent_window) >= 3 and sum(a in FIRE_ACTIONS for a in recent_window) == 0:
        if proposed_action == 2:
            return 4
        if proposed_action == 3:
            return 5

    # If the model keeps repeating pure FIRE for a while, allow a tiny lane adjustment.
    if len(recent_window) == 4 and all(a == 1 for a in recent_window) and proposed_action == 1:
        counts = Counter(recent_actions[-8:])
        return 4 if counts[4] <= counts[5] else 5

    return proposed_action


def choose_action_openai(
    client: OpenAI,
    model: str,
    frames: List[np.ndarray],
    system_prompt: str,
    recent_history: List[Tuple[int, float]],
    temperature: float,
) -> int:
    tiled = make_tiled_image(frames)
    image_url = pil_to_data_url(tiled)
    recent_actions = [a for a, _ in recent_history]

    user_text = (
        'Choose the best NEXT action for Atari Space Invaders.\n'
        'Use the newest frame most heavily and older frames only for motion inference.\n'
        'Recent local history:\n'
        f'{summarize_history(recent_history)}\n'
        'Return JSON only.'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': user_text},
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                ],
            },
        ],
        temperature=temperature,
        max_tokens=48,
        response_format={'type': 'json_object'},
    )
    text = response.choices[0].message.content or ''
    proposed_action = extract_action(text)
    return apply_action_guardrails(proposed_action, recent_actions)


def play_episode(
    client: OpenAI,
    model: str,
    system_prompt: str,
    max_steps: int = 2000,
    record_frames: bool = False,
    temperature: float = 0.0,
):
    env = VLMAtariWrapper()
    frames, _ = env.reset()
    total_reward = 0.0
    decision_frames = []
    recent_history: Deque[Tuple[int, float]] = deque(maxlen=8)

    for _ in range(max_steps):
        action = choose_action_openai(
            client=client,
            model=model,
            frames=frames,
            system_prompt=system_prompt,
            recent_history=list(recent_history),
            temperature=temperature,
        )
        frames, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        recent_history.append((action, reward))
        if record_frames:
            decision_frames.extend(frames)
        if terminated or truncated:
            break

    env.close()
    return decision_frames, total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--api-base', type=str, default='http://localhost:11434/v1')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--prompt-file', type=str, default='prompt.txt')
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    if args.api_base == 'none':
        raise NotImplementedError(
            'This reference solution only supports OpenAI-compatible API mode. '
            'Serve a Qwen model behind a chat-completions endpoint and pass --api-base.'
        )

    client = OpenAI(base_url=args.api_base, api_key=os.environ.get('OPENAI_API_KEY', 'EMPTY'))
    system_prompt = load_prompt(args.prompt_file)

    rewards = []
    for i in range(1, args.episodes + 1):
        _, reward = play_episode(
            client,
            args.model,
            system_prompt,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )
        rewards.append(reward)
        print(f'Episode {i}: reward={reward:.1f}')

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    print(f'Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}')


if __name__ == '__main__':
    main()
