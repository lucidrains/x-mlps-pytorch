# /// script
# dependencies = [
#   "x-mlps-pytorch",
#   "torch",
#   "einops",
#   "einx==0.4.0",
#   "x-ppo",
#   "gymnasium[box2d]",
#   "moviepy",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "mean-conc-beta==0.2.0",
#   "fire"
# ]
#
# [tool.uv.sources]
# x-mlps-pytorch = { path = "." }
# ///

from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path

import fire
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import Module
from torch.optim import Adam

from mean_conc_beta import Beta as MeanConcBeta
from x_ppo import ppo_actor_loss

from x_mlps_pytorch import AttnResidualNormedMLP

# running observation normalizer

class RunningObsNormalizer(Module):
    def __init__(self, dim, clip = 10.):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim, dtype = torch.float64))
        self.register_buffer('var', torch.ones(dim, dtype = torch.float64))
        self.register_buffer('count', torch.tensor(1e-4, dtype = torch.float64))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x = torch.as_tensor(x, dtype = torch.float64).reshape(-1, self.mean.shape[0])

        batch_mean, batch_var = x.mean(dim = 0), x.var(dim = 0, unbiased = False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean.copy_(self.mean + delta * batch_count / total)
        self.var.copy_((self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total) / total)
        self.count.copy_(total)

    def forward(self, x):
        x = torch.as_tensor(x, dtype = torch.float32)
        return ((x - self.mean.float()) / (self.var.float() + 1e-8).sqrt()).clamp(-self.clip, self.clip)

# actor critic

class Actor(Module):
    def __init__(self, state_dim, action_dim, hidden_dim, bounds, depth = 12, lora_rank = 16, num_streams = 1):
        super().__init__()

        self.net = AttnResidualNormedMLP(
            dim = hidden_dim,
            depth = depth,
            dim_in = state_dim,
            dim_out = hidden_dim,
            use_rmsnorm = True,
            lora_rank = lora_rank,
            num_streams = num_streams,
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )

        self.dist_module = MeanConcBeta(
            bounds = bounds,
            init_conc = 2.,
            unimodal = True,
            eps = 1e-5,
            detach_entropy_mean = True,
        )

    def forward(self, x):
        out = self.action_head(self.net(x))
        return self.dist_module(rearrange(out, '... (a c) -> ... a c', c = 2))

    @torch.no_grad()
    def act(self, x):
        dist = self.forward(x)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim = -1)

class Critic(Module):
    def __init__(self, state_dim, hidden_dim, depth = 12, lora_rank = 16, num_streams = 1):
        super().__init__()

        self.net = AttnResidualNormedMLP(
            dim = hidden_dim,
            depth = depth,
            dim_in = state_dim,
            dim_out = hidden_dim,
            use_rmsnorm = True,
            lora_rank = lora_rank,
            num_streams = num_streams,
        )

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.value_head(self.net(x)).squeeze(-1)

    @torch.no_grad()
    def forward_eval(self, x):
        return self.forward(x)

# rollouts

def run_episode(actor, env, normalizer, device, seed = None, deterministic = True):
    obs, _ = env.reset(seed = seed)

    trajectory = []
    episode_reward = 0.
    done = False

    while not done:
        with torch.no_grad():
            dist = actor(normalizer(obs).float().to(device))
            action = (dist.mean if deterministic else dist.sample()).clamp(-1., 1.)

        env_action = action.cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(env_action)

        trajectory.append(dict(state = obs.tolist(), action = env_action.tolist(), reward = float(reward)))
        episode_reward += float(reward)
        obs = next_obs
        done = terminated or truncated

    lander = getattr(env.unwrapped, 'lander', None)

    return dict(
        reward = episode_reward,
        steps = len(trajectory),
        landed_safely = bool(lander is not None and not lander.awake),
        trajectory = trajectory,
    )

def evaluate(actor, normalizer, seeds, device):
    env = gym.make('LunarLander-v3', continuous = True)
    results = [dict(seed = seed, **run_episode(actor, env, normalizer, device, seed = seed)) for seed in seeds]
    env.close()
    return results

def record_video(actor, normalizer, device, seed, video_path, output_dir):
    tmp_dir = output_dir / '_tmp_video'
    shutil.rmtree(str(tmp_dir), ignore_errors = True)
    tmp_dir.mkdir(parents = True, exist_ok = True)

    env = gym.make('LunarLander-v3', continuous = True, render_mode = 'rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder = str(tmp_dir), name_prefix = 'lunarlander', episode_trigger = lambda episode: True)

    result = run_episode(actor, env, normalizer, device, seed = seed)
    env.close()

    videos = sorted(tmp_dir.glob('*.mp4'))

    if videos:
        shutil.copy(str(videos[0]), str(video_path))
        result['video'] = str(video_path)

    shutil.rmtree(str(tmp_dir), ignore_errors = True)

    return result

# plots

def plot_training(episode_rewards, rolling_avgs, out_path, target_avg):
    fig, ax = plt.subplots(figsize = (10, 5), dpi = 150)

    ax.plot(episode_rewards, alpha = 0.35, color = '#93c5fd', linewidth = 0.9, label = 'Episode Return')
    ax.plot(rolling_avgs, color = '#1d4ed8', linewidth = 2.4, label = 'Rolling Avg (20 eps)')
    ax.axhline(0., color = 'gray', linestyle = ':', alpha = 0.7)
    ax.axhline(target_avg, color = '#16a34a', linestyle = '--', linewidth = 1.6, label = f'Target (+{target_avg:.0f})')
    ax.axhline(200., color = '#dc2626', linestyle = '--', linewidth = 1.2, alpha = 0.7, label = 'Solved (+200)')

    ax.set_title('Continuous LunarLander-v3: PPO with AttnResidualNormedMLP (depth 12)', fontsize = 12, fontweight = 'bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.legend(loc = 'lower right')
    ax.grid(True, linestyle = '--', alpha = 0.4)

    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()

def plot_trajectory(result, out_path):
    states = np.array([step['state'] for step in result['trajectory']])
    actions = np.array([step['action'] for step in result['trajectory']])
    rewards = np.array([step['reward'] for step in result['trajectory']])

    fig, axes = plt.subplots(2, 2, figsize = (13, 9), dpi = 150)

    ax = axes[0][0]
    ax.plot(states[:, 0], states[:, 1], color = '#1d4ed8', linewidth = 1.8)
    ax.scatter(states[0, 0], states[0, 1], color = '#16a34a', s = 80, zorder = 5, label = 'start')
    ax.scatter(states[-1, 0], states[-1, 1], color = '#dc2626', s = 80, zorder = 5, label = 'end')
    ax.add_patch(plt.Polygon([[-0.22, -0.22], [0.22, -0.22], [0.22, 0.], [-0.22, 0.]], color = '#facc15', alpha = 0.6, label = 'landing pad'))
    ax.set_title(f'Lander trajectory | return {result["reward"]:+.1f} | {result["steps"]} steps | safe landing {result["landed_safely"]}')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_aspect('equal', adjustable = 'datalim')
    ax.legend(loc = 'best')
    ax.grid(True, linestyle = '--', alpha = 0.4)

    ax = axes[0][1]
    ax.plot(actions[:, 0], color = '#2563eb', linewidth = 1.6, label = 'main engine')
    ax.plot(actions[:, 1], color = '#f97316', linewidth = 1.6, label = 'side engines')
    ax.set_title('Actions over time')
    ax.set_xlabel('step')
    ax.legend(loc = 'best')
    ax.grid(True, linestyle = '--', alpha = 0.4)

    ax = axes[1][0]
    state_names = ('horizontal velocity', 'vertical velocity', 'angle', 'angular velocity')
    state_colors = ('#65a30d', '#dc2626', '#7c3aed', '#0891b2')

    for index, (name, color) in enumerate(zip(state_names, state_colors)):
        ax.plot(states[:, index + 2], color = color, linewidth = 1.4, label = name)
    ax.set_title('Lander state over time')
    ax.set_xlabel('step')
    ax.legend(loc = 'best', fontsize = 8)
    ax.grid(True, linestyle = '--', alpha = 0.4)

    ax = axes[1][1]
    ax.plot(np.cumsum(rewards), color = '#16a34a', linewidth = 2.)
    ax.set_title('Cumulative reward')
    ax.set_xlabel('step')
    ax.set_ylabel('return')
    ax.grid(True, linestyle = '--', alpha = 0.4)

    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()

# training

def train(
    max_episodes = 1000,
    target_avg = 50.,
    n_steps = 1024,
    epochs = 4,
    minibatch_size = 64,
    lr = 3e-4,
    gamma = 0.999,
    lam = 0.98,
    clip = 0.2,
    ent_coef = 0.01,
    reward_scale = 0.01,
    max_grad_norm = 0.5,
    depth = 12,
    hidden_dim = 64,
    lora_rank = 16,
    num_streams = 1,
    rolling_window_size = 20,
    eval_every = 25,
    device = 'cpu',
    seed = 42,
    output_dir = './recordings/lunar_attn_res',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    print('=' * 90)
    print('PPO on continuous LunarLander-v3 with AttnResidualNormedMLP')
    print(f'  depth {depth} | hidden {hidden_dim} | lora rank {lora_rank} | streams {num_streams} | device {device}')
    print(f'  target rolling avg +{target_avg:.0f} within {max_episodes} episodes | artifacts -> {output_dir}')
    print('=' * 90, flush = True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('LunarLander-v3', continuous = True)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    bounds = np.stack([env.action_space.low, env.action_space.high], axis = -1)

    normalizer = RunningObsNormalizer(state_dim)
    actor = Actor(state_dim, action_dim, hidden_dim, bounds, depth = depth, lora_rank = lora_rank, num_streams = num_streams).to(device)
    critic = Critic(state_dim, hidden_dim, depth = depth, lora_rank = lora_rank, num_streams = num_streams).to(device)

    opt_actor = Adam(actor.parameters(), lr = lr, betas = (0.9, 0.99))
    opt_critic = Adam(critic.parameters(), lr = lr, betas = (0.9, 0.99))

    obs, _ = env.reset(seed = seed)
    normalizer.update(obs)

    episode_rewards, rolling_avgs = [], []
    rolling_rewards = deque(maxlen = rolling_window_size)
    best_eval = -float('inf')
    target_hit = False
    total_steps = 0
    update = 0

    pbar = tqdm(total = max_episodes, desc = 'episodes')

    while len(episode_rewards) < max_episodes and not target_hit:
        update += 1

        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        episode_reward = 0.

        for _ in range(n_steps):
            x = normalizer(obs)
            obs_buf.append(x)

            x = x.float().to(device)
            action, log_prob = actor.act(x)
            value = critic.forward_eval(x)

            env_action = action.clamp(-1., 1.).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            act_buf.append(action.cpu().numpy())
            logp_buf.append(float(log_prob))
            rew_buf.append(float(reward) * reward_scale)
            done_buf.append(bool(done))
            val_buf.append(float(value))

            episode_reward += float(reward)
            total_steps += 1

            obs = next_obs
            normalizer.update(obs)

            if done:
                episode_rewards.append(episode_reward)
                rolling_rewards.append(episode_reward)
                rolling_avgs.append(float(np.mean(rolling_rewards)))
                pbar.update(1)

                if len(rolling_rewards) >= rolling_window_size and rolling_avgs[-1] >= target_avg:
                    target_hit = True
                    print(f'converged: rolling avg {rolling_avgs[-1]:+.2f} >= +{target_avg:.0f} at episode {len(episode_rewards) - 1} ({total_steps} env steps)', flush = True)

                episode_reward = 0.
                obs, _ = env.reset()
                normalizer.update(obs)

            if len(episode_rewards) >= max_episodes or target_hit:
                break

        rewards = torch.tensor(rew_buf, dtype = torch.float32, device = device)
        values = torch.tensor(val_buf, dtype = torch.float32, device = device)
        dones = torch.tensor(done_buf, dtype = torch.float32, device = device)
        last_value = float(critic.forward_eval(normalizer(obs).float().to(device)))

        advantages = torch.zeros(len(rew_buf), dtype = torch.float32, device = device)
        last_gae = 0.

        for timestep in reversed(range(len(rew_buf))):
            if timestep == len(rew_buf) - 1:
                next_nonterminal, next_value = 1. - dones[timestep], last_value
            else:
                next_nonterminal, next_value = 1. - dones[timestep + 1], values[timestep + 1]

            delta = rewards[timestep] + gamma * next_value * next_nonterminal - values[timestep]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[timestep] = last_gae

        returns = advantages + values

        b_obs = torch.from_numpy(np.stack(obs_buf)).float().to(device)
        b_act = torch.from_numpy(np.stack(act_buf)).float().to(device)
        b_logp = torch.tensor(logp_buf, dtype = torch.float32, device = device)

        actor.train()
        critic.train()

        for _ in range(epochs):
            indices = np.random.permutation(len(rew_buf))

            for start in range(0, len(rew_buf), minibatch_size):
                mb = indices[start:start + minibatch_size]

                dist = actor(b_obs[mb])
                new_logp = dist.log_prob(b_act[mb]).sum(dim = -1)
                entropy = dist.entropy().sum(dim = -1)

                policy_loss = ppo_actor_loss(new_logp, b_logp[mb], advantages[mb], clip, normalize_advantages = True).mean()
                actor_loss = policy_loss - ent_coef * entropy.mean()

                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt_actor.step()

                value_loss = 0.5 * ((critic(b_obs[mb]) - returns[mb]) ** 2).mean()

                opt_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                opt_critic.step()

        pbar.set_postfix(ep_rew = f'{episode_rewards[-1]:+.1f}', rolling = f'{rolling_avgs[-1]:+.1f}')

        if update % eval_every == 0 or target_hit:
            results = evaluate(actor, normalizer, seeds = range(1000, 1005), device = device)
            eval_avg = float(np.mean([result['reward'] for result in results]))
            landings = sum(int(result['landed_safely']) for result in results)
            print(f'update {update:4d} | episode {len(episode_rewards):4d} | eval avg {eval_avg:+7.2f} ({landings}/5 safe landings)', flush = True)

            if eval_avg > best_eval:
                best_eval = eval_avg
                best = max(results, key = lambda result: result['reward'])
                record_video(actor, normalizer, device, best['seed'], output_dir / 'best.mp4', output_dir)

            plot_training(episode_rewards, rolling_avgs, output_dir / 'training_rewards.png', target_avg)

    env.close()
    pbar.close()

    plot_training(episode_rewards, rolling_avgs, output_dir / 'training_rewards.png', target_avg)

    final_results = evaluate(actor, normalizer, seeds = range(2000, 2010), device = device)
    final_rewards = [result['reward'] for result in final_results]
    print(f'final eval: mean {np.mean(final_rewards):+.2f} | best {np.max(final_rewards):+.2f} | safe landings {sum(int(result["landed_safely"]) for result in final_results)}/10', flush = True)

    best = max(final_results, key = lambda result: result['reward'])
    if not best['landed_safely']:
        best = next((result for result in final_results if result['landed_safely']), best)

    video_result = record_video(actor, normalizer, device, best['seed'], output_dir / 'final_eval.mp4', output_dir)

    if video_result['landed_safely']:
        shutil.copy(str(output_dir / 'final_eval.mp4'), str(output_dir / 'successful_landing.mp4'))

    plot_trajectory(video_result, output_dir / 'successful_trajectory.png')

    with open(output_dir / 'successful_trajectory.json', 'w') as file:
        json.dump(video_result, file, indent = 2)

    with open(output_dir / 'metrics.json', 'w') as file:
        json.dump(dict(
            episodes = len(episode_rewards),
            env_steps = total_steps,
            target_avg = target_avg,
            converged = target_hit,
            final_rolling_avg = rolling_avgs[-1],
            best_eval_avg = best_eval,
            final_eval = dict(
                mean = float(np.mean(final_rewards)),
                best = float(np.max(final_rewards)),
                safe_landings = int(sum(int(result['landed_safely']) for result in final_results)),
                episodes = final_results,
            ),
        ), file, indent = 2)

    torch.save(dict(actor = actor.state_dict(), critic = critic.state_dict(), normalizer = normalizer.state_dict()), str(output_dir / 'agent.pt'))

    print(f'saved training curve, trajectory plot, videos and agent to {output_dir}')
    return dict(converged = target_hit, episodes = len(episode_rewards), best_eval_avg = best_eval)


if __name__ == '__main__':
    fire.Fire(train)
