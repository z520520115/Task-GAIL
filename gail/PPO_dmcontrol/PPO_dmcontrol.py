import matplotlib.pyplot as plt
import numpy as np
import tqdm
from dm_control import suite
from dm_control import viewer
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

env = suite.load(domain_name="swimmer", task_name="swimmer6")
action_spec = env.action_spec()
t = env.reset()
state_size = sum(len(x) for x in t.observation.values())

# print(state_size, action_spec.shape[0]) # 25 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Visualize episode
def to_state(time_step):
    state = np.concatenate([x for x in time_step.observation.values()])
    return state

def policy(time_step):
    state = to_state(time_step)
    action = ppo.select_action(state, None)
    return action
viewer.launch(env, policy=policy)
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, hidden1=64, hidden2=32):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)
        self.apply(init_weights)

    def forward(self, x):
        pass

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if memory:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std=0.5, gamma=0.99, K_epochs=80, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.value_coef = 0.5
        self.entropy_coef = 0.001

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.policy.act(state, memory)
        action = action.cpu().numpy().flatten()
        action = np.clip(action, -1.0, 1.0)
        return action

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Form a batch
        rewards = torch.tensor(rewards).to(device)
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluate old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Find the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Normalize the advantages
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surr_loss = - torch.min(surr1, surr2)  # minus for maximizing surrogate loss

            value_loss = self.value_coef * (state_values - rewards).pow(2).mean()
            entropy_loss = self.entropy_coef * dist_entropy

            # Total loss
            loss = surr_loss + value_loss - entropy_loss

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

class VecNormalizer(object):
    def __init__(self, obs_shape):
        self.ob_mean = np.zeros(obs_shape)
        self.m2 = np.zeros(obs_shape)
        self.epsilon = 1e-8
        self.n = 0

    @property
    def ob_var(self):
        if self.n <= 1:
            return 0
        else:
            return self.m2 / self.n

    def update(self, obs):
        self.n += 1

        if self.n == 1:
            self.ob_mean = obs
            self.m2 = 0
        else:
            new_mean = self.ob_mean + (obs - self.ob_mean) / self.n
            self.m2 += (obs - self.ob_mean) * (obs - new_mean);
            self.ob_mean = new_mean

    def __call__(self, time_step, skip_normalization=True):
        obs = np.concatenate([x for x in time_step.observation.values()])

        if not skip_normalization:
            self.update(obs)
            obs = (obs - self.ob_mean) / np.sqrt(self.ob_var + self.epsilon)
            obs = np.clip(obs, -10., 10.)

        reward = time_step.reward
        done = time_step.last()
        return obs, reward, done

ppo = PPO(state_size, action_spec.shape[0])
memory = Memory()
normalizer = VecNormalizer(state_size)

time_step = 0
update_timestep = 4096
max_t = 1000

max_episodes = 500
log_interval = 20

scores = []
avg_dist = 0
avg_length = 0
viewer.launch(env, policy=policy)
for i_episode in range(1, max_episodes + 1):
    env_step = env.reset()
    state, _, _ = normalizer(env_step)

    score = 0
    last_reward = None
    dist0 = np.linalg.norm(env_step.observation["to_target"])

    for t in range(max_t):
        time_step += 1

        # Select action and make a step
        action = ppo.select_action(state, memory)
        env_step = env.step(action)
        state, reward, done = normalizer(env_step)

        # Calculate reward: the difference between this and last timesteps' rewards
        delta = 0 if last_reward is None else reward - last_reward
        last_reward = reward
        # Add encouragement for reaching target
        bonus = 0
        if reward == 1.0:  # check if reached the target
            bonus = 1000 - t  # add a bonus
            done = True  # stop episode early

        memory.rewards.append(1000 * delta + bonus)
        memory.is_terminals.append(done)

        # Update every 4096 timesteps
        if time_step % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()

        score += reward
        if done:
            break

    # Calculate travelled ratio
    dist1 = np.linalg.norm(env_step.observation["to_target"])
    avg_dist += (dist0 - dist1) / dist0

    scores.append(score + (999 - t))  # add the score we would get if we didn't stop episode early
    avg_length += t

    # logging
    if i_episode % log_interval == 0:
        avg_dist /= log_interval
        avg_length = int(avg_length / log_interval)

        print("Episode {} \t Dist: {:.4f} \t Env score: {:.2f} \t Len: {}".format(i_episode, avg_dist,
                                                                                  np.mean(scores[-20:]), avg_length))
        avg_dist = 0
        avg_length = 0

    if i_episode % 500 == 0:
        torch.save(ppo.policy.state_dict(), "../PPO_dmcontrol/PPO_swimmer_delta_{}.pth".format(i_episode))

torch.save(ppo.policy.state_dict(), "../PPO_dmcontrol/PPO_swimmer_delta.pth")


viewer.launch(env, policy=policy)