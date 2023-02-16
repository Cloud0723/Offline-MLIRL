import copy
import os

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from tqdm import tqdm

from utils import ReplayPoolCtxt, ReplayPool, FasterReplayPool, FasterReplayPoolCtxt, TanhTransform, Transition, TransitionContext, filter_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def stable_network_forward(self, x):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        return mu, logstd

    def compute_action(self, mu, std, get_logprob=False):
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action, logprob, mean

    def forward(self, x, get_logprob=False):
        mu, logstd = self.stable_network_forward(x)
        std = logstd.exp()
        return self.compute_action(mu, std, get_logprob)


class DoubleQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class SAC_Agent:

    def __init__(self, seed, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256,
                 update_interval=1, buffer_size=1e6, target_entropy=None, augment_sac=False, rad_rollout=False,
                 context_type='rad_augmentation'):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy else -action_dim / 2
        self.batchsize = batchsize
        self.update_interval = update_interval

        torch.manual_seed(seed)

        # context-sac
        self.augment_sac = augment_sac
        self.rad_rollout = rad_rollout
        self.context_type = context_type

        original_state_dim = state_dim

        if self.augment_sac:
            if context_type == 'rad_augmentation':
                print('Augmenting state vector with context_type={}.'.format(context_type))
                state_dim *= 2
            elif context_type == 'rad_magnitude':
                state_dim += 1

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)

        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        if augment_sac and rad_rollout:
            # self.replay_pool = ReplayPoolCtxt(capacity=int(buffer_size))
            self.replay_pool = FasterReplayPoolCtxt(action_dim=action_dim, state_dim=original_state_dim, capacity=int(buffer_size))
        else:
            # self.replay_pool = ReplayPool(capacity=int(buffer_size))
            self.replay_pool = FasterReplayPool(action_dim=action_dim, state_dim=original_state_dim, capacity=int(buffer_size))

    def reallocate_replay_pool(self, new_size: int):
        assert new_size != self.replay_pool.capacity, "Error, you've tried to allocate a new pool which has the same length"
        new_replay_pool = FasterReplayPoolCtxt(self.replay_pool._action_dim, self.replay_pool._action_dim, capacity=new_size)
        new_replay_pool.initialise(self.replay_pool)
        self.replay_pool = new_replay_pool

    def get_action(self, state, state_filter=None, deterministic=False, oac=False):
        if state_filter:
            state = state_filter(state)
        state = torch.Tensor(state).view(1, -1).to(device)
        if oac:
            action, _, mean = self._get_optimistic_action(state)
        else:
            with torch.no_grad():
                action, _, mean = self.policy(state)
        if deterministic:
            return np.atleast_1d(mean.squeeze().cpu().numpy())
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def _get_optimistic_action(self, state, get_logprob=False):

        beta_UB = 4.66  # Table 1: https://arxiv.org/pdf/1910.12807.pdf
        delta = 23.53  # Table 1: https://arxiv.org/pdf/1910.12807.pdf

        mu, logvar = self.policy.stable_network_forward(state)
        mu.requires_grad_()
        std = logvar.exp()

        action = torch.tanh(mu)
        q_1, q_2 = self.q_funcs(state, action)

        mu_Q = (q_1 + q_2) / 2.0

        sigma_Q = torch.abs(q_1 - q_2) / 2.0

        Q_UB = mu_Q + beta_UB * sigma_Q

        grad = torch.autograd.grad(Q_UB, mu)
        grad = grad[0]

        grad = grad.detach()
        mu = mu.detach()
        std = std.detach()

        Sigma_T = torch.pow(std.detach(), 2)
        denom = torch.sqrt(
            torch.sum(torch.mul(torch.pow(grad, 2), Sigma_T))) + 10e-6

        # Obtain the change in mu
        mu_C = math.sqrt(2.0 * delta) * torch.mul(Sigma_T, grad) / denom

        mu_E = mu + mu_C

        assert mu_E.shape == std.shape

        # dist = TanhNormal(mu_E, std)
        # action = dist.sample()

        return self.policy.compute_action(mu_E, std, get_logprob=get_logprob)

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.log_alpha.exp() * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def optimize(self, n_updates, state_filter=None, env_pool=None, env_ratio=0.05, augment_data=False,reward_function=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0

        hide_progress = True if n_updates < 50 else False

        for i in tqdm(range(n_updates), disable=hide_progress, ncols=100):
            if env_pool and env_ratio != 0:
                n_env_samples = int(env_ratio * self.batchsize)
                n_model_samples = self.batchsize - n_env_samples
                env_samples = env_pool.sample(n_env_samples)._asdict()
                model_samples = self.replay_pool.sample(n_model_samples)._asdict()
                if self.augment_sac and self.rad_rollout:
                    samples = TransitionContext(*[env_samples[key] + model_samples[key] for key in env_samples])
                else:
                    samples = Transition(*[env_samples[key] + model_samples[key] for key in env_samples])
            else:
                samples = self.replay_pool.sample(self.batchsize)
            #print(len(samples),samples)
            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)

            if self.augment_sac and self.rad_rollout:
                # Concatenate the context with the state after filtering, this is done on model before
                rad_batch = torch.FloatTensor(samples.rad_context).to(device)
                state_batch = torch.cat((state_batch, rad_batch), 1)
                nextstate_batch = torch.cat((nextstate_batch, rad_batch), 1)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            if reward_function:
                #print('before:',reward_batch)
                reward_batch += reward_function(torch.cat((state_batch,action_batch),1))
                #print('after:',reward_batch)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)

            if augment_data:
                # Delta context
                magnitude = 0.5
                high = 1 + magnitude
                low = 1 - magnitude
                scale = high - low

                # Direct nextstate augmentation
                # # magnitude = np.random.uniform(0, 0.5)
                # random_amplitude_scaling = (torch.rand(state_batch.shape) * scale + low).to(device)
                # # state_batch *= random_amplitude_scaling
                # nextstate_batch *= random_amplitude_scaling

                # random_amplitude_scaling = (torch.rand(state_batch.shape[0]) * scale + low).unsqueeze(1).to(device)
                random_amplitude_scaling = (torch.rand(state_batch.shape) * scale + low).to(device)
                delta_batch = nextstate_batch - state_batch
                delta_batch *= random_amplitude_scaling
                nextstate_batch = state_batch + delta_batch

                # Additive Noise
                # random_amplitude_scaling = torch.randn_like(state_batch) * 0.1
                # nextstate_batch += random_amplitude_scaling

                if self.augment_sac and not self.rad_rollout and self.context_type == 'rad_augmentation':
                    state_batch = torch.cat((state_batch, random_amplitude_scaling), 1)
                    nextstate_batch = torch.cat((nextstate_batch, random_amplitude_scaling), 1)
                elif self.augment_sac and not self.rad_rollout and self.context_type == 'rad_magnitude':
                    state_batch = torch.cat((state_batch, magnitude * torch.ones(state_batch.shape[0], 1).to(device)), 1)
                    nextstate_batch = torch.cat((nextstate_batch, magnitude * torch.ones(state_batch.shape[0], 1).to(device)), 1)

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch,
                                                                 nextstate_batch, done_batch)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.q_funcs.parameters():
                p.requires_grad = False
            pi_loss_step, a_loss_step = self.update_policy_and_temp(state_batch)
            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()
            for p in self.q_funcs.parameters():
                p.requires_grad = True

            self.alpha = self.log_alpha.exp()

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()
            pi_loss += pi_loss_step.detach().item()
            a_loss += a_loss_step.detach().item()
            if i % self.update_interval == 0:
                self.update_target()
        return q1_loss, q2_loss, pi_loss, a_loss

    def save_policy(self, save_path, num_epochs, rew=None):
        q_funcs, target_q_funcs, policy, log_alpha = self.q_funcs, self.target_q_funcs, self.policy, self.log_alpha

        if rew is None:
            save_path = os.path.join(save_path, "torch_policy_weights_{}_epochs.pt".format(num_epochs))
        else:
            save_path = os.path.join(save_path, "torch_policy_weights_{}_epochs_{}.pt".format(num_epochs, rew))

        torch.save({
            'double_q_state_dict': q_funcs.state_dict(),
            'target_double_q_state_dict': target_q_funcs.state_dict(),
            'policy_state_dict': policy.state_dict(),
            'log_alpha_state_dict': log_alpha
        }, save_path)
