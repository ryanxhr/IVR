import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_multiplier=1.0, log_std_offset=-1.0):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state):
        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Double_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Double_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class CQL(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            eta=0.005,
            alpha=5.0,
            n_actions=10,
            importance_sample=False,
            temp=1.0,
            cql_clip_diff_min=-np.inf,
            cql_clip_diff_max=np.inf,
            entropy_weight=0.0,
            use_automatic_entropy_tuning=False,
    ):

        self.policy = Actor(state_dim, action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.critic = Double_Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if use_automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_entropy_weight = torch.zeros(1, requires_grad=True, device=device)
            self.entropy_weight_optimizer = torch.optim.Adam([self.log_entropy_weight], lr=3e-4)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.alpha = alpha
        self.n_actions = n_actions
        self.importance_sample = importance_sample
        self.temp = temp
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.entropy_weight = entropy_weight
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        self.discount = discount
        self.eta = eta
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.policy(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, _, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, _, _ = self.policy(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        current_Q1, current_Q2 = self.critic(state, action)

        td_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # CQL
        state_repeat = torch.repeat_interleave(state, self.n_actions, 0)
        next_state_repeat = torch.repeat_interleave(next_state, self.n_actions, 0)

        cql_random_actions = action.new_empty((batch_size*self.n_actions, self.action_dim), requires_grad=False).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis, _ = self.policy(state_repeat)
        cql_next_actions, cql_next_log_pis, _ = self.policy(next_state_repeat)
        cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
        cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

        cql_q1_current_actions, cql_q2_current_actions = self.critic(state_repeat, cql_current_actions)
        cql_q1_next_actions, cql_q2_next_actions = self.critic(next_state_repeat, cql_next_actions)
        cql_q1_rand, cql_q2_rand = self.critic(state_repeat, cql_random_actions)

        cql_q1_current_actions = cql_q1_current_actions.reshape(-1, self.n_actions)
        cql_q2_current_actions = cql_q2_current_actions.reshape(-1, self.n_actions)
        cql_q1_next_actions = cql_q1_next_actions.reshape(-1, self.n_actions)
        cql_q2_next_actions = cql_q2_next_actions.reshape(-1, self.n_actions)
        cql_q1_rand = cql_q1_rand.reshape(-1, self.n_actions)
        cql_q2_rand = cql_q2_rand.reshape(-1, self.n_actions)

        cql_current_log_pis = cql_current_log_pis.reshape(-1, self.n_actions)
        cql_next_log_pis = cql_next_log_pis.reshape(-1, self.n_actions)

        cql_cat_q1 = torch.cat([cql_q1_rand, cql_q1_next_actions, cql_q1_current_actions], 1)
        cql_cat_q2 = torch.cat([cql_q2_rand, cql_q2_next_actions, cql_q2_current_actions], 1)

        if self.importance_sample:
            random_density = np.log(0.5 ** self.action_dim)
            cql_cat_q1 = torch.cat(
                [cql_q1_rand - random_density,
                 cql_q1_next_actions - cql_next_log_pis.detach(),
                 cql_q1_current_actions - cql_current_log_pis.detach()],
                dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand - random_density,
                 cql_q2_next_actions - cql_next_log_pis.detach(),
                 cql_q2_current_actions - cql_current_log_pis.detach()],
                dim=1
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.temp, dim=1, keepdim=True) * self.temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.temp, dim=1, keepdim=True) * self.temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - current_Q1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
            ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - current_Q2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
            ).mean()

        cql_loss = self.alpha * (cql_qf1_diff + cql_qf2_diff)
        q_loss = td_loss + cql_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Compute policy loss
        pi, log_pi, _ = self.policy(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.squeeze(torch.min(qf1_pi, qf2_pi))
        p_loss = (self.entropy_weight * log_pi - min_qf_pi).mean()

        # Optimize the policy
        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()

        # Optimize the entropy weight
        if self.use_automatic_entropy_tuning:
            entropy_loss = -(self.log_entropy_weight * (log_pi + self.target_entropy).detach()).mean()

            self.entropy_weight_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_weight_optimizer.step()

            self.entropy_weight = self.log_entropy_weight.exp()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.eta * param.data + (1 - self.eta) * target_param.data)

    def train_mix(self, replay_buffer_e, replay_buffer_b, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state_e, action_e, next_state_e, _, reward_e, not_done_e = replay_buffer_e.sample(batch_size)
        state_b, action_b, next_state_b, _, reward_b, not_done_b = replay_buffer_b.sample(batch_size)

        with torch.no_grad():
            next_action, _, _ = self.policy(next_state_b)
            target_Q1, target_Q2 = self.critic_target(next_state_b, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_b + not_done_b * self.discount * target_Q
        current_Q1, current_Q2 = self.critic(state_b, action_b)

        td_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # CQL
        state_repeat = torch.repeat_interleave(state_b, self.n_actions, 0)
        next_state_repeat = torch.repeat_interleave(next_state_b, self.n_actions, 0)

        cql_random_actions = action_b.new_empty((batch_size*self.n_actions, self.action_dim), requires_grad=False).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis, _ = self.policy(state_repeat)
        cql_next_actions, cql_next_log_pis, _ = self.policy(next_state_repeat)
        cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
        cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

        cql_q1_current_actions, cql_q2_current_actions = self.critic(state_repeat, cql_current_actions)
        cql_q1_next_actions, cql_q2_next_actions = self.critic(next_state_repeat, cql_next_actions)
        cql_q1_rand, cql_q2_rand = self.critic(state_repeat, cql_random_actions)

        cql_q1_current_actions = cql_q1_current_actions.reshape(-1, self.n_actions)
        cql_q2_current_actions = cql_q2_current_actions.reshape(-1, self.n_actions)
        cql_q1_next_actions = cql_q1_next_actions.reshape(-1, self.n_actions)
        cql_q2_next_actions = cql_q2_next_actions.reshape(-1, self.n_actions)
        cql_q1_rand = cql_q1_rand.reshape(-1, self.n_actions)
        cql_q2_rand = cql_q2_rand.reshape(-1, self.n_actions)

        cql_current_log_pis = cql_current_log_pis.reshape(-1, self.n_actions)
        cql_next_log_pis = cql_next_log_pis.reshape(-1, self.n_actions)

        cql_cat_q1 = torch.cat([cql_q1_rand, cql_q1_next_actions, cql_q1_current_actions], 1)
        cql_cat_q2 = torch.cat([cql_q2_rand, cql_q2_next_actions, cql_q2_current_actions], 1)

        if self.importance_sample:
            random_density = np.log(0.5 ** self.action_dim)
            cql_cat_q1 = torch.cat(
                [cql_q1_rand - random_density,
                 cql_q1_next_actions - cql_next_log_pis.detach(),
                 cql_q1_current_actions - cql_current_log_pis.detach()],
                dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand - random_density,
                 cql_q2_next_actions - cql_next_log_pis.detach(),
                 cql_q2_current_actions - cql_current_log_pis.detach()],
                dim=1
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.temp, dim=1, keepdim=True) * self.temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.temp, dim=1, keepdim=True) * self.temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - current_Q1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
            ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - current_Q2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
            ).mean()

        cql_loss = self.alpha * (cql_qf1_diff + cql_qf2_diff)
        q_loss = td_loss + cql_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Compute policy loss
        pi, log_pi, _ = self.policy(state_e)
        qf1_pi, qf2_pi = self.critic(state_e, pi)
        min_qf_pi = torch.squeeze(torch.min(qf1_pi, qf2_pi))
        p_loss = (self.entropy_weight * log_pi - min_qf_pi).mean()

        # Optimize the policy
        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()

        # Optimize the entropy weight
        if self.use_automatic_entropy_tuning:
            entropy_loss = -(self.log_entropy_weight * (log_pi + self.target_entropy).detach()).mean()

            self.entropy_weight_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_weight_optimizer.step()

            self.entropy_weight = self.log_entropy_weight.exp()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.eta * param.data + (1 - self.eta) * target_param.data)

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_policy"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))
