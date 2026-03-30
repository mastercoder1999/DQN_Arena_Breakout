import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import time
import os

# ============================================================
# ENV FIXES
# ============================================================
os.environ["SDL_AUDIODRIVER"] = "dummy"
device = torch.device("cpu")

# ============================================================
# DQN NETWORK (DeepMind)
# ============================================================
class DQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        return self.net(x / 255.0)

# ============================================================
# UTILS
# ============================================================
def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

def force_fire(env, n=6):
    obs = None
    for _ in range(n):
        obs, _, _, _, _ = env.step(1)
    return obs

# ============================================================
# REPLAY BUFFER
# ============================================================
class ReplayBuffer:
    def __init__(self, size=1_000_000):
        self.buffer = deque(maxlen=size)

    def add(self, *exp):
        self.buffer.append(exp)

    def sample(self, batch):
        batch = random.sample(self.buffer, batch)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.stack(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
# TRAINING (DOUBLE DQN)
# ============================================================
def train(episodes=600):
    env = gym.make("ALE/Breakout-v5", frameskip=4)
    action_dim = env.action_space.n

    policy = DQN(action_dim).to(device)
    target = DQN(action_dim).to(device)
    target.load_state_dict(policy.state_dict())

    opt = optim.Adam(policy.parameters(), lr=2.5e-4)
    buffer = ReplayBuffer()

    epsilon = 1.0
    eps_min = 0.02
    eps_decay = 5_000_000

    gamma = 0.99
    batch_size = 32
    update_target_every = 30_000
    step = 0

    for ep in range(episodes):
        obs, info = env.reset()
        obs = force_fire(env)
        lives = info["lives"]

        frame = preprocess(obs)
        stack = np.stack([frame]*4)

        total_reward = 0

        while True:
            step += 1

            # EXPLORATION (no NOOP early)
            if random.random() < epsilon or step < 50_000:
                action = random.choice([1,2,3])
            else:
                with torch.no_grad():
                    q = policy(torch.from_numpy(stack).float().unsqueeze(0))
                action = q.argmax(1).item()

            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward

            # Reward shaping (survival)
            reward += 0.01

            # Re-fire after death
            if info["lives"] < lives:
                obs = force_fire(env)
            lives = info["lives"]

            new_frame = preprocess(obs)
            new_stack = np.concatenate([stack[1:], new_frame[None]], axis=0)

            buffer.add(stack, action, reward, new_stack, term or trunc)
            stack = new_stack

            # TRAIN
            if len(buffer) > 50_000:
                s,a,r,ns,d = buffer.sample(batch_size)

                q = policy(s).gather(1,a[:,None]).squeeze()

                with torch.no_grad():
                    next_a = policy(ns).argmax(1)
                    q_next = target(ns).gather(1,next_a[:,None]).squeeze()

                target_q = r + gamma * q_next * (1-d)

                loss = nn.MSELoss()(q, target_q)

                opt.zero_grad()
                loss.backward()
                opt.step()

            # UPDATE TARGET
            if step % update_target_every == 0:
                target.load_state_dict(policy.state_dict())

            epsilon = max(eps_min, epsilon - 1/eps_decay)

            if term or trunc:
                break

        print(f"Episode {ep:4d} | Score {total_reward:.1f} | ε={epsilon:.3f}")

    torch.save(policy.state_dict(), "dqn_model.pt")
    env.close()

# ============================================================
# PLAY (RENDER)
# ============================================================
def play():
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="rgb_array")
    policy = DQN(env.action_space.n).to(device)
    policy.load_state_dict(torch.load("dqn_model.pt"))
    policy.eval()

    obs,_ = env.reset()
    obs = force_fire(env)
    frame = preprocess(obs)
    stack = np.stack([frame]*4)

    lives = 5
    reward_sum = 0

    while True:
        with torch.no_grad():
            action = policy(torch.from_numpy(stack).float().unsqueeze(0)).argmax(1).item()

        obs, r, term, trunc, info = env.step(action)
        reward_sum += r

        if info["lives"] < lives:
            obs = force_fire(env)
        lives = info["lives"]

        display = cv2.resize(obs, (640,480), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Breakout - Improved DQN", display[:,:,::-1])

        if cv2.waitKey(1) == ord("q"):
            break

        frame = preprocess(obs)
        stack = np.concatenate([stack[1:], frame[None]], axis=0)

        if term or trunc:
            obs,_ = env.reset()
            obs = force_fire(env)
            frame = preprocess(obs)
            stack = np.stack([frame]*4)

    env.close()
    cv2.destroyAllWindows()
    print("Final score:", reward_sum)

# ============================================================
if __name__ == "__main__":
    train()
    #play()
