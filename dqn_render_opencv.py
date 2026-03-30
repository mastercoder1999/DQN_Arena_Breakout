import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm
import cv2
import time
import os

# ============================================================
# DISABLE AUDIO (prevents ALSA spam)
# ============================================================
os.environ["SDL_AUDIODRIVER"] = "dummy"

# ============================================================
# DEVICE AUTO-SELECTION (CPU fallback for MX350)
# ============================================================
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU detected → capability sm_{major}{minor}")

    if major < 7:
        print("⚠ GPU too old for modern PyTorch → using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
else:
    print("No GPU detected → using CPU")
    device = torch.device("cpu")

print("Using device:", device)

# ============================================================
# DEEPMIND CNN ARCHITECTURE
# ============================================================
class DQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        return self.net(x / 255.0)

# ============================================================
# PREPROCESSING
# ============================================================
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame
# Force fire
def force_fire(env, n=5):
    """Force FIRE a few times to reliably start Breakout"""
    obs = None
    for _ in range(n):
        obs, _, _, _, _ = env.step(1)  # action FIRE
    return obs

# ============================================================
# REPLAY BUFFER
# ============================================================
class ReplayBuffer:
    def __init__(self, size=1_000_000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch):
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch))
        return (
            np.stack(s),
            np.array(a),
            np.array(r),
            np.stack(ns),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
# TRAINING LOOP
# ============================================================
def train_dqn(env_name="ALE/Breakout-v5", episodes=300):
    env = gym.make(env_name, frameskip=4)
    action_dim = env.action_space.n

    policy = DQN(action_dim).to(device)
    target = DQN(action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=2.5e-4)
    buffer = ReplayBuffer()

    gamma = 0.99
    batch_size = 32
    update_target_every = 10_000
    epsilon_min = 0.1
    epsilon_decay = 1_000_000

    # === DEFAULT VALUES (will be overwritten if checkpoint exists)
    epsilon = 1.0
    step_count = 0
    start_episode = 0

    # ============================================================
    # LOAD CHECKPOINT IF EXISTS
    # ============================================================
    if os.path.exists("dqn_checkpoint.pt"):
        print("✅ Loading checkpoint...")
        ckpt = torch.load("dqn_checkpoint.pt", map_location=device, weights_only=False)

        policy.load_state_dict(ckpt["policy"])
        target.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])

        epsilon = ckpt["epsilon"]
        step_count = ckpt["step_count"]
        buffer.buffer = ckpt["replay_buffer"]

        start_episode = ckpt["episode"] + 1
        print(f"↪ Resuming from episode {start_episode}")

    else:
        target.load_state_dict(policy.state_dict())

    def stack_frames(s, f):
        return np.append(f[1:], np.expand_dims(s, 0), axis=0)

    print("\n=== TRAINING START ===\n")

    for ep in range(start_episode, start_episode + episodes):
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(1)  # FIRE

        frame = preprocess(obs)
        frames = np.stack([frame] * 4)
        total_reward = 0

        for _ in range(8000):
            step_count += 1

            # === ACTION ===
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy(
                        torch.tensor(frames, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    )
                action = torch.argmax(q).item()

            new_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            new_frame = preprocess(new_obs)
            new_frames = stack_frames(new_frame, frames)

            buffer.add(frames, action, reward, new_frames, done)
            frames = new_frames
            total_reward += reward

            # === TRAIN ===
            if len(buffer) > 50_000:
                s, a, r, ns, d = buffer.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32).to(device)
                ns = torch.tensor(ns, dtype=torch.float32).to(device)
                a = torch.tensor(a, dtype=torch.int64).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)

                q = policy(s).gather(1, a.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    q_next = target(ns).max(1)[0]
                    target_q = r + gamma * q_next * (1 - d)

                loss = nn.MSELoss()(q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # === TARGET UPDATE ===
            if step_count % update_target_every == 0:
                target.load_state_dict(policy.state_dict())

            epsilon = max(epsilon_min, epsilon - (1.0 / epsilon_decay))

            if done:
                break

        print(f"Episode {ep} | Reward = {total_reward:.1f} | ε = {epsilon:.3f}")

        # ============================================================
        # SAVE CHECKPOINT EACH EPISODE
        # ============================================================
        checkpoint = {
            "policy": policy.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon,
            "step_count": step_count,
            "episode": ep,
            "replay_buffer": buffer.buffer,
        }

        torch.save(checkpoint, "dqn_checkpoint.pt")
        torch.save(policy.state_dict(), "dqn_model.pt")

    env.close()

# ============================================================
# PLAY WITH OPENCV RENDERING
# ============================================================
def play_with_render(env_name="ALE/Breakout-v5"):
    print("Loading trained model...")

    # frameskip=1 → rendu plus fluide / logique plus simple
    env = gym.make(env_name, frameskip=1, render_mode="rgb_array")
    action_dim = env.action_space.n

    policy = DQN(action_dim).to(device)
    policy.load_state_dict(torch.load("dqn_model.pt", map_location=device))
    policy.eval()

    print("Starting game with OpenCV rendering (Q to quit)")

    # === RESET & INITIAL SERVE ===
    obs, info = env.reset()
    obs = force_fire(env, n=8)

    frame = preprocess(obs)
    frames = np.stack([frame] * 4)

    lives = info.get("lives", 5)

    def stack_frames(s, f):
        return np.append(f[1:, :, :], np.expand_dims(s, 0), axis=0)

    total_reward = 0
    step = 0

    while True:
        step += 1

        # === DQN ACTION ===
        with torch.no_grad():
            q = policy(
                torch.tensor(frames, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
        action = torch.argmax(q).item()

        # === STEP ENV ===
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # === LIFE LOSS DETECTION ===
        current_lives = info.get("lives", lives)
        if current_lives < lives:
            # lost a life → re-serve ball
            obs = force_fire(env, n=8)

        lives = current_lives

        # === RENDER (lightweight, stable) ===
        if step % 2 == 0:
            display = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            display = cv2.resize(display, (640, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Breakout - DQN (OpenCV)", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # === UPDATE STATE STACK ===
        processed = preprocess(obs)
        frames = stack_frames(processed, frames)

        # === TRUE GAME OVER ===
        if terminated or truncated:
            obs, info = env.reset()
            obs = force_fire(env, n=8)
            frame = preprocess(obs)
            frames = np.stack([frame] * 4)
            lives = info.get("lives", lives)

    env.close()
    cv2.destroyAllWindows()
    print("Final score:", total_reward)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # TRAIN:
    #train_dqn(episodes=100)

    # PLAY:
    play_with_render()
