# RL Training Examples

## PPO with Stable-Baselines3

Training a PPO agent on the Ant environment.

```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from zeno.gym import make_vec

# Create training and evaluation environments
train_envs = make_vec("ant", num_envs=64)
eval_envs = make_vec("ant", num_envs=16)

# Configure PPO
model = PPO(
    "MlpPolicy",
    train_envs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/",
)

# Evaluation callback
eval_callback = EvalCallback(
    eval_envs,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
)

# Train
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
)

# Save final model
model.save("ppo_ant_final")

# Cleanup
train_envs.close()
eval_envs.close()
```

## Custom Training Loop

A minimal PPO implementation for educational purposes.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from zeno.gym import make_vec

# Hyperparameters
NUM_ENVS = 64
NUM_STEPS = 2048
NUM_UPDATES = 1000
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
EPOCHS = 10
BATCH_SIZE = 256

# Create environment
envs = make_vec("ant", num_envs=NUM_ENVS)
obs_dim = envs.observation_space.shape[1]
act_dim = envs.action_space.shape[1]

# Simple actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(256, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(256, 1)

    def forward(self, obs):
        features = self.shared(obs)
        return self.actor_mean(features), self.critic(features)

    def get_action(self, obs):
        mean, value = self(obs)
        std = self.actor_logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs, actions):
        mean, value = self(obs)
        std = self.actor_logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy

model = ActorCritic(obs_dim, act_dim)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
obs, _ = envs.reset()
obs = torch.tensor(obs, dtype=torch.float32)

for update in range(NUM_UPDATES):
    # Collect rollout
    rollout_obs = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_dones = []
    rollout_values = []

    for step in range(NUM_STEPS):
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs)

        rollout_obs.append(obs)
        rollout_actions.append(action)
        rollout_log_probs.append(log_prob)
        rollout_values.append(value)

        # Step environment
        next_obs, rewards, terminateds, truncateds, infos = envs.step(action.numpy())
        dones = np.logical_or(terminateds, truncateds)

        rollout_rewards.append(torch.tensor(rewards, dtype=torch.float32))
        rollout_dones.append(torch.tensor(dones, dtype=torch.float32))

        obs = torch.tensor(next_obs, dtype=torch.float32)

    # Compute GAE
    with torch.no_grad():
        _, _, last_value = model.get_action(obs)
        advantages = torch.zeros(NUM_STEPS, NUM_ENVS)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                next_value = last_value
            else:
                next_value = rollout_values[t + 1]
            delta = rollout_rewards[t] + GAMMA * next_value * (1 - rollout_dones[t]) - rollout_values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * (1 - rollout_dones[t]) * lastgaelam

        returns = advantages + torch.stack(rollout_values)

    # Flatten rollout
    b_obs = torch.stack(rollout_obs).reshape(-1, obs_dim)
    b_actions = torch.stack(rollout_actions).reshape(-1, act_dim)
    b_log_probs = torch.stack(rollout_log_probs).reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    # Normalize advantages
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    # PPO update
    total_loss = 0
    for epoch in range(EPOCHS):
        indices = torch.randperm(NUM_STEPS * NUM_ENVS)
        for start in range(0, NUM_STEPS * NUM_ENVS, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_indices = indices[start:end]

            new_log_probs, new_values, entropy = model.evaluate(
                b_obs[batch_indices],
                b_actions[batch_indices]
            )

            # Policy loss
            ratio = (new_log_probs - b_log_probs[batch_indices]).exp()
            pg_loss1 = -b_advantages[batch_indices] * ratio
            pg_loss2 = -b_advantages[batch_indices] * torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * ((new_values - b_returns[batch_indices]) ** 2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = pg_loss + 0.5 * v_loss + 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

    # Logging
    mean_reward = torch.stack(rollout_rewards).sum(0).mean().item()
    print(f"Update {update}: mean_reward={mean_reward:.2f}, loss={total_loss/(EPOCHS * (NUM_STEPS * NUM_ENVS // BATCH_SIZE)):.4f}")

envs.close()

# Save model
torch.save(model.state_dict(), "ppo_ant.pt")
```

## SAC Training

Soft Actor-Critic with Zeno.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from zeno.gym import make_vec

# Hyperparameters
NUM_ENVS = 16
BUFFER_SIZE = 100000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # Entropy coefficient

# Networks
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, obs):
        features = self.net(obs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1)

# Create environment
envs = make_vec("ant", num_envs=NUM_ENVS)
obs_dim = envs.observation_space.shape[1]
act_dim = envs.action_space.shape[1]

# Initialize networks
policy = PolicyNetwork(obs_dim, act_dim)
q1 = QNetwork(obs_dim, act_dim)
q2 = QNetwork(obs_dim, act_dim)
q1_target = QNetwork(obs_dim, act_dim)
q2_target = QNetwork(obs_dim, act_dim)
q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())

policy_opt = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LEARNING_RATE)

# Replay buffer
buffer = deque(maxlen=BUFFER_SIZE)

# Training
obs, _ = envs.reset()

for step in range(100000):
    # Sample action
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action, _ = policy.sample(obs_tensor)
        action = action.numpy()

    # Step environment
    next_obs, rewards, terminateds, truncateds, infos = envs.step(action)
    dones = np.logical_or(terminateds, truncateds)

    # Store transitions
    for i in range(NUM_ENVS):
        buffer.append((obs[i], action[i], rewards[i], next_obs[i], dones[i]))

    obs = next_obs

    # Update
    if len(buffer) > BATCH_SIZE:
        batch = random.sample(buffer, BATCH_SIZE)
        b_obs = torch.tensor([t[0] for t in batch], dtype=torch.float32)
        b_action = torch.tensor([t[1] for t in batch], dtype=torch.float32)
        b_reward = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        b_next_obs = torch.tensor([t[3] for t in batch], dtype=torch.float32)
        b_done = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        # Q-function update
        with torch.no_grad():
            next_action, next_log_prob = policy.sample(b_next_obs)
            q1_next = q1_target(b_next_obs, next_action).squeeze()
            q2_next = q2_target(b_next_obs, next_action).squeeze()
            q_next = torch.min(q1_next, q2_next) - ALPHA * next_log_prob
            q_target = b_reward + GAMMA * (1 - b_done) * q_next

        q1_loss = ((q1(b_obs, b_action).squeeze() - q_target) ** 2).mean()
        q2_loss = ((q2(b_obs, b_action).squeeze() - q_target) ** 2).mean()

        q_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        q_opt.step()

        # Policy update
        new_action, log_prob = policy.sample(b_obs)
        q_new = torch.min(q1(b_obs, new_action), q2(b_obs, new_action)).squeeze()
        policy_loss = (ALPHA * log_prob - q_new).mean()

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # Target update
        for param, target_param in zip(q1.parameters(), q1_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(q2.parameters(), q2_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    if step % 1000 == 0:
        print(f"Step {step}")

envs.close()
```

## Evaluation Script

Evaluating a trained policy.

```python
import numpy as np
import torch
from zeno.gym import make_vec

def evaluate_policy(model_path, env_name, num_episodes=100):
    """Evaluate a trained policy."""
    envs = make_vec(env_name, num_envs=num_episodes)

    # Load model (adjust based on your model type)
    model = torch.load(model_path)
    model.eval()

    obs, _ = envs.reset()
    episode_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)
    done_mask = np.zeros(num_episodes, dtype=bool)

    while not done_mask.all():
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = model.get_action(obs_tensor, deterministic=True)

        obs, rewards, terminateds, truncateds, infos = envs.step(action.numpy())
        dones = np.logical_or(terminateds, truncateds)

        # Update statistics for non-done episodes
        active = ~done_mask
        episode_rewards[active] += rewards[active]
        episode_lengths[active] += 1

        done_mask |= dones

    envs.close()

    print(f"Evaluation Results ({num_episodes} episodes):")
    print(f"  Mean reward: {episode_rewards.mean():.2f} ± {episode_rewards.std():.2f}")
    print(f"  Mean length: {episode_lengths.mean():.1f} ± {episode_lengths.std():.1f}")
    print(f"  Min reward: {episode_rewards.min():.2f}")
    print(f"  Max reward: {episode_rewards.max():.2f}")

    return episode_rewards, episode_lengths

# Usage
# evaluate_policy("ppo_ant.pt", "ant", num_episodes=100)
```
