import itertools
import random
import math
from collections import deque, namedtuple
from typing import List
import time
import sys
from copy import deepcopy
import pickle
import datetime
import pdb
import time

import gym
from gym.wrappers import TimeLimit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
import math

from utils import MeanStdevFilter, prepare_data, reward_func, GaussianMSELoss, truncated_normal_init, Transition, \
    ReplayPool, check_or_make_folder, FasterReplayPool

### Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnsembleGymEnv(gym.Env):
    """Wraps the Ensemble with a gym API, Outputs Normal states, and contains a copy of the true environment"""

    def __init__(self, params, env, eval_env, due_override=None):
        super(EnsembleGymEnv, self).__init__()
        self.name = params['env_name']
        self.real_env = env
        self.eval_env = eval_env
        self.observation_space = self.real_env.observation_space
        self.action_space = self.real_env.action_space
        params['action_space'] = self.real_env.action_space.shape[0]
        params['observation_space'] = self.real_env.observation_space.shape[0]
        self.model = Ensemble(params, due_override=due_override)
        self.current_state = self.reset()
        self.reward_head = params['reward_head']
        self.reward_func = reward_func
        self.action_bounds = self.get_action_bounds()
        self.spec = self.real_env.spec
        self._elapsed_steps = 0
        self._max_timesteps = self.real_env.env.spec.max_episode_steps
        torch.manual_seed(params['seed'])

    def get_action_bounds(self):
        Bounds = namedtuple('Bounds', ('lowerbound', 'upperbound'))
        lb = self.real_env.action_space.low
        ub = self.real_env.action_space.high
        return Bounds(lowerbound=lb, upperbound=ub)

    def seed(self, seed=None):
        return self.real_env.seed(seed)

    def train_model(self, max_epochs, n_samples: int = 200000, d4rl_init=False, save_model=False,
                    min_model_epochs=None):
        self.model.train_model(
            max_epochs=max_epochs,
            n_samples=n_samples,
            d4rl_init=d4rl_init,
            save_model=save_model,
            min_model_epochs=min_model_epochs)

    def step(self, action):
        action = np.clip(action, self.action_bounds.lowerbound, self.action_bounds.upperbound)
        next_state, reward = self.model.predict_state(
            self.current_state.reshape(1, -1),
            action.reshape(1, -1))
        if not reward:
            reward = self.reward_func(
                self.current_state,
                next_state,
                action,
                self.name,
                is_done_func=self.model.is_done_func)
        if self._elapsed_steps > self._max_timesteps:
            done = True
        elif self.model.is_done_func:
            done = self.model.is_done_func(torch.Tensor(next_state).reshape(1, -1)).item()
        else:
            done = False
        self.current_state = next_state
        self._elapsed_steps += 1
        return next_state, reward, done, {}

    def reset(self):
        self.current_state = self.eval_env.reset()
        self._elapsed_steps = 0
        return self.current_state

    def render(self, mode='human'):
        raise NotImplementedError

    def update_state_filter(self, new_state):
        self.model.state_filter.update(new_state)

    def update_action_filter(self, new_action):
        self.model.action_filter.update(new_action)

    def convert_filter_to_torch(self):
        self.model.state_filter.update_torch()
        self.model.action_filter.update_torch()


class TransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, batch: Transition, state_filter, action_filter):
        state_action_filtered, delta_filtered = prepare_data(
            batch.state,
            batch.action,
            batch.nextstate,
            state_filter,
            action_filter)
        self.data_X = torch.Tensor(state_action_filtered)
        self.data_y = torch.Tensor(delta_filtered)
        self.data_r = torch.Tensor(np.array(batch.reward))

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index], self.data_r[index]


class EnsembleTransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, batch: Transition, state_filter, action_filter, n_models=1):
        state_action_filtered, delta_filtered = prepare_data(
            batch.state,
            batch.action,
            batch.nextstate,
            state_filter,
            action_filter)
        data_count = state_action_filtered.shape[0]
        # idxs = np.arange(0,data_count)[None,:].repeat(n_models, axis=0)
        # [np.random.shuffle(row) for row in idxs]
        idxs = np.random.randint(data_count, size=[n_models, data_count])
        self._n_models = n_models
        self.data_X = torch.Tensor(state_action_filtered[idxs])
        self.data_y = torch.Tensor(delta_filtered[idxs])
        self.data_r = torch.Tensor(np.array(batch.reward)[idxs])

    def __len__(self):
        return self.data_X.shape[1]

    def __getitem__(self, index):
        return self.data_X[:, index], self.data_y[:, index], self.data_r[:, index]


class Ensemble(object):
    def __init__(self, params, due_override=None):

        self.params = params
        self.models = {i: Model(input_dim=params['ob_dim'] + params['ac_dim'],
                                output_dim=params['ob_dim'] + params['reward_head'],
                                is_probabilistic=params['logvar_head'],
                                is_done_func=params['is_done_func'],
                                reward_head=params['reward_head'],
                                seed=params['seed'] + i,
                                l2_reg_multiplier=params['l2_reg_multiplier'],
                                num=i)
                       for i in range(params['num_models'])}
        self.elites = {i: self.models[i] for i in range(params['num_elites'])}
        self._elites_idx = list(range(params['num_elites']))
        self.num_models = params['num_models']
        self.num_elites = params['num_elites']
        self.output_dim = params['ob_dim'] + params['reward_head']
        self.ob_dim = params['ob_dim']
        self.memory = FasterReplayPool(action_dim=params['ac_dim'], state_dim=params['ob_dim'],
                                       capacity=params['train_memory'])
        self.memory_val = FasterReplayPool(action_dim=params['ac_dim'], state_dim=params['ob_dim'],
                                           capacity=params['val_memory'])
        self.train_val_ratio = params['train_val_ratio']
        self.is_done_func = params['is_done_func']
        self.is_probabilistic = params['logvar_head']
        self._model_lr = params['model_lr'] if 'model_lr' in params else 0.001
        weights = [weight for model in self.models.values() for weight in model.weights]
        if self.is_probabilistic:
            self.max_logvar = torch.full((self.output_dim,), 0.5, requires_grad=True, device=device)
            self.min_logvar = torch.full((self.output_dim,), -10.0, requires_grad=True, device=device)
            weights.append({'params': [self.max_logvar]})
            weights.append({'params': [self.min_logvar]})
            self.set_model_logvar_limits()
        self.optimizer = torch.optim.Adam(weights, lr=self._model_lr)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.3, verbose=False)
        self._model_id = "Model_{}_seed{}_{}".format(params['env_name'], params['seed'],
                                                     datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S'))
        self._env_name = params['env_name']
        self.state_filter = MeanStdevFilter(params['observation_space'])
        self.action_filter = MeanStdevFilter(params['action_space'])
        self.use_automatic_lam_tuning = params['tune_mopo_lam']

        self._removed_models = []

        if params['mopo']:
            # this value is ignored if tuning is used
            self.mopo_lam = params['mopo_lam']

            if self.use_automatic_lam_tuning:
                self.target_uncertainty = params['mopo_uncertainty_target']
                self.log_mopo_lam = torch.full((1,), 1.0, requires_grad=True, device=device)
                self.lam_optimizer = torch.optim.Adam([self.log_mopo_lam], lr=0.01)
        else:
            self.mopo_lam = 0
        self.due_override = due_override

        self._morel_thresh = None
        self._morel_halt_reward = 0

    # def reset_models(self):
    #     params = self.params
    #     self.models = {i:Model(input_dim=params['ob_dim'] + params['ac_dim'],
    #                     output_dim=params['ob_dim'] + params['reward_head'],
    #                     is_probabilistic = params['logvar_head'],
    #                     is_done_func = params['is_done_func'],
    #                     reward_head = params['reward_head'],
    #                     seed = params['seed'] + i,
    #                     num=i)
    #             for i in range(params['num_models'])}

    def forward(self, x: torch.Tensor):
        model_index = int(np.random.uniform() * len(self.models.keys()))
        return self.models[model_index].forward(x)

    def predict_state(self, state: np.array, action: np.array) -> (np.array, float):
        model_index = int(np.random.uniform() * len(self.models.keys()))
        return self.models[model_index].predict_state(state, action, self.state_filter, self.action_filter)

    def predict_state_at(self, state: np.array, action: np.array, model_index: int) -> (np.array, float):
        return self.models[model_index].predict_state(state, action, self.state_filter, self.action_filter)

    def add_data(self, step: Transition):
        # for step in rollout:
        self.memory.push(step)

    def add_data_validation(self, step: Transition):
        # for step in rollout:
        self.memory_val.push(step)

    def get_next_states_rewards(self, state, action, get_var=False, deterministic=False, return_mean=False):
        if self.mopo_lam == 0:
            nextstates_rewards = [
                elite.get_next_state_reward(state, action, self.state_filter, self.action_filter, get_var,
                                            deterministic, return_mean) for elite
                in self.elites.values()]
        else:
            nextstates_rewards = [
                model.get_next_state_reward(state, action, self.state_filter, self.action_filter, get_var,
                                            deterministic, return_mean) for model
                in self.models.values()]
        nextstates_all = torch.stack([torch.cat(nsr[0], dim=1) if get_var else nsr[0] for nsr in nextstates_rewards])
        rewards_all = torch.stack([torch.cat(nsr[1], dim=1) if get_var else nsr[1] for nsr in nextstates_rewards])
        return nextstates_all, rewards_all

    def get_max_state_action_uncertainty(self, state, action):
        state_action_uncertainties = [
            elite.get_state_action_uncertainty(state, action, self.state_filter, self.action_filter) for
            elite in self.elites.values()]
        return torch.max(torch.stack([torch.norm(logvar.exp(), p='fro') for logvar in state_action_uncertainties]))

    #TODO: Use this method in the rollout code
    def get_elites_prediction(self, state, action):
        """ Gets a next state prediction according to the Elites """

        allocation = torch.randint(0, self.num_elites, (state.shape[0],), device=device)

        return_var = True

        allocation = torch.tensor([self._elites_idx[idx] for idx in allocation]).to(device)
        # needs to get logvar for MOPO penalty
        get_var = True

        allocation_states = allocation.repeat(self.ob_dim, 1).T.view(1, -1, self.ob_dim)
        allocation_rewards = allocation.view(1, -1, 1)

        # need the parametric mean of the next states for the LOMPO and M2AC uncertainty metrics
        return_mean = True

        with torch.no_grad():

            nextstates_all, rewards_all = self.get_next_states_rewards(state, action, get_var, False, return_mean)

            nextstates_all, nextstates_all_mu, nextstates_logvar = nextstates_all.chunk(3, dim=2)
            rewards_all, rewards_all_mu, rewards_logvar = rewards_all.chunk(3, dim=2)
            nextstates = nextstates_all.gather(0, allocation_states).squeeze()
            rewards = rewards_all.gather(0, allocation_rewards).squeeze()

        return {'nextstates': nextstates, 'nextstates_all': nextstates_all, 'nextstates_logvar': nextstates_logvar, 'rewards': rewards, 'rewards_all': rewards_all, 'rewards_logvar': rewards_logvar, 'nextstates_mu': nextstates_all_mu, 'rewards_mu': rewards_all_mu, 'allocation': allocation}

    def get_all_penalties_state_action(self, state, action, nextstates_info:dict=None):
        """Gets all the penalties (and predicted next state) from a given state-action tuple"""
        penalty_list = ['mopo_paper', 'morel', 'lompo', 'm2ac', 'ensemble_var', 'ensemble_std']

        if not nextstates_info:
            nextstates_info = self.get_elites_prediction(state, action)

        with torch.no_grad():
            penalties = {}
        
            for p in penalty_list:
                penalties[p] = self.get_penalty(p, **nextstates_info).cpu().numpy()

        return penalties, nextstates_info['nextstates']

    def set_morel_hparams(self, morel_threshold, morel_halt_reward):
        self._morel_thresh, self._morel_halt_reward = morel_threshold, morel_halt_reward
        print("Set MOReL penalty threshold to {}, and negative HALT reward to {}".format(self._morel_thresh, self._morel_halt_reward))

    def get_penalty(self, penalty_name: str, nextstates, nextstates_all, nextstates_logvar, rewards, rewards_all, rewards_logvar, nextstates_mu, rewards_mu, allocation):
        """Gets the penalty depending on name"""

        if penalty_name == 'mopo_default':
            nextstates_std = nextstates_logvar.exp().sqrt()
            mopo_pen = nextstates_std.norm(2,2).amax(0)
        elif penalty_name == 'mopo_paper':
            all_std = torch.cat((nextstates_logvar, rewards_logvar), dim=2).exp().sqrt()
            mopo_pen = all_std.norm(2,2).amax(0)
        elif penalty_name == 'ensemble_var':
            nextstates_var = nextstates_logvar.exp()
            mean_of_vars = torch.mean(nextstates_var, dim=0)
            var_of_means = torch.var(nextstates_all, dim=0)
            vr = mean_of_vars + var_of_means
            mopo_pen = torch.mean(vr, dim=1)
        elif penalty_name == 'ensemble_std':
            nextstates_var = nextstates_logvar.exp()
            mean_of_vars = torch.mean(nextstates_var, dim=0)
            var_of_means = torch.var(nextstates_all, dim=0)
            std = (mean_of_vars + var_of_means).sqrt()
            mopo_pen = torch.mean(std, dim=1)
        elif penalty_name == 'ensemble_var_rew':
            rewards_var = rewards_logvar.exp()
            mean_of_vars = torch.mean(rewards_var, dim=0)
            var_of_means = torch.var(rewards_all, dim=0)
            vr = mean_of_vars + var_of_means
            mopo_pen = torch.mean(vr, dim=1)
        elif penalty_name == 'ensemble_var_comb':
            nextstates_var = nextstates_logvar.exp()
            mean_of_vars = torch.mean(nextstates_var, dim=0)
            var_of_means = torch.var(nextstates_all, dim=0)
            vr = mean_of_vars + var_of_means
            mopo_pen1 = torch.mean(vr, dim=1)
            rewards_var = rewards_logvar.exp()
            mean_of_vars = torch.mean(rewards_var, dim=0)
            var_of_means = torch.var(rewards_all, dim=0)
            vr = mean_of_vars + var_of_means
            mopo_pen2 = torch.mean(vr, dim=1)
            mopo_pen = mopo_pen1 + mopo_pen2
        elif penalty_name == 'morel':
            # first try naive nested for loop
            # mopo_pen = torch.zeros_like(rewards)
            # for i, ns_i in enumerate(nextstates_all):
            #     for j, ns_j  in enumerate(nextstates_all):
            #         # only need upper right triangle
            #         if j > i:
            #             mopo_pen = torch.max(mopo_pen, (ns_i - ns_j).norm(2,1))

            # this parallelises the above code, runs 10x faster, and is 10x less readable
            mopo_pen = torch.cdist(nextstates_all.swapaxes(0,1), nextstates_all.swapaxes(0,1), 2).amax((1,2))
        elif penalty_name == 'lompo':
            # Several steps:
            # 1) Get the next states and rewards from models
            # 2) Now sample a random next state from the uniform
            # 3) For the next state, pass it through the Gaussian induced by each NN to get a log likelihood
            # 4) Now take the variance of that likelihood over the models for the sampled next state
            # 5) Average over each reward, i.e.,: \hat{r} = 1/K * \sum_{i=1}^K [r_i(s_t,a_t)] - \lambda * u_i(s_t,a_t)
            # Let's not average reward for now to keep it consistent; we just care about uncertainty metric
            # rewards = rewards_all.mean(0)
            mus, stds = nextstates_mu, nextstates_logvar.exp().sqrt()
            dist = torch.distributions.Normal(mus, stds)
            ll = dist.log_prob(nextstates).sum(2)
            # the penalty is then just the variance of the log likelihoods, averaged across each next state prediction
            mopo_pen = ll.var(0)
        elif penalty_name == 'm2ac':
            # Several steps:
            # Let's assume they only did this on the Elites, but no implementation so can't be sure
            # 1) Figure out which model was allocated, and what elites remain
            # 2) Merge the remaining model Gaussians, leaving out the model that was allocated
            # 3) Calculate the KL divergence between the model that was allocated, and the merged Gaussian from step 2
            # We did 1) already, now for...
            # Merging the remaining Gaussians
            allocation_states = allocation.repeat(self.ob_dim, 1).T.view(1, -1, self.ob_dim)
            allocation_rewards = allocation.view(1, -1, 1)
            # Need to come up with allocations that:
            # 1) Exclude non-elites # actually scrap this
            # 2) Exclude the allocated model so we can do OvR metric
            # TODO: Maybe this is inefficient, is there some way to vectorize?
            # exclude_dic = {a: torch.tensor([idx for idx in self._elites_idx if idx != a]).to(device) for a in self._elites_idx}
            exclude_dic = {a: torch.tensor([idx for idx in range(self.num_models) if idx != a]).to(device) for a in range(self.num_models)}
            allocation_ovr = torch.stack([exclude_dic[i] for i in allocation.cpu().numpy()])
            allocation_ovr_states = allocation_ovr.unsqueeze(2).repeat(1,1,self.ob_dim*2).swapaxes(1,0)
            allocation_ovr_rewards = allocation_ovr.unsqueeze(2).repeat(1,1,2).swapaxes(1,0)
            nextstates_mu_logvar = torch.cat((nextstates_mu, nextstates_logvar), dim=2)
            rewards_mu_logvar = torch.cat((rewards_mu, rewards_logvar), dim=2)
            nextstates_mu_alloc, nextstates_logvar_alloc = nextstates_mu_logvar.gather(0, allocation_states.repeat(1,1,2)).squeeze(0).chunk(2, dim=1)
            nextstates_mu_ovr, nextstates_logvar_ovr = nextstates_mu_logvar.gather(0, allocation_ovr_states).squeeze(0).chunk(2, dim=2)
            rewards_mu_alloc, rewards_logvar_alloc = rewards_mu_logvar.gather(0, allocation_rewards.repeat(1,1,2)).squeeze(0).chunk(2, dim=1)
            rewards_mu_ovr, rewards_logvar_ovr = rewards_mu_logvar.gather(0, allocation_ovr_rewards).squeeze(0).chunk(2, dim=2)
            nsr_mu_alloc = torch.cat((nextstates_mu_alloc, rewards_mu_alloc), dim=1)
            nsr_logvar_alloc = torch.cat((nextstates_logvar_alloc, rewards_logvar_alloc), dim=1)
            nsr_mu_ovr = torch.cat((nextstates_mu_ovr, rewards_mu_ovr), dim=2)
            nsr_logvar_ovr = torch.cat((nextstates_logvar_ovr, rewards_logvar_ovr), dim=2)
            merge_mu = nsr_mu_ovr.mean(0)
            merge_std = ((nsr_logvar_ovr.exp() + nsr_mu_ovr**2).mean(0) - merge_mu**2).clamp(min=1e-8).sqrt()
            alloc_gaussian = torch.distributions.Normal(nsr_mu_alloc, nsr_logvar_alloc.exp().sqrt())
            merge_gaussian = torch.distributions.Normal(merge_mu, merge_std)
            mopo_pen = torch.distributions.kl_divergence(alloc_gaussian, merge_gaussian).sum(1)
        else:
            raise NotImplementedError

        return mopo_pen

    def random_env_step(self, state, action, get_var=False, deterministic=False, disable_mopo=False,IRL = False):
        """Randomly allocate the data through the different dynamics models"""

        allocation = torch.randint(0, self.num_elites, (state.shape[0],), device=device)

        return_var = get_var

        if self.mopo_lam != 0 or self._morel_thresh:
            # converts elite index into all-models index
            allocation = torch.tensor([self._elites_idx[idx] for idx in allocation]).to(device)
            # needs to get logvar for MOPO penalty
            get_var = True

        allocation_states = allocation.repeat(self.ob_dim, 1).T.view(1, -1, self.ob_dim)
        allocation_rewards = allocation.view(1, -1, 1)

        # need the parametric mean of the next states for the LOMPO and M2AC uncertainty metrics
        return_mean = True if self.params['mopo_penalty_type'] in ['lompo', 'm2ac'] else False

        nextstates_all, rewards_all = self.get_next_states_rewards(state, action, get_var, deterministic, return_mean)

        if get_var:
            if return_mean:
                nextstates_all, nextstates_all_mu, nextstates_logvar = nextstates_all.chunk(3, dim=2)
                rewards_all, rewards_all_mu, rewards_logvar = rewards_all.chunk(3, dim=2)
            else:
                nextstates_all, nextstates_logvar = nextstates_all.chunk(2, dim=2)
                rewards_all, rewards_logvar = rewards_all.chunk(2, dim=2)
                nextstates_all_mu = None
                rewards_all_mu = None
        nextstates = nextstates_all.gather(0, allocation_states).squeeze(0)
        rewards = rewards_all.gather(0, allocation_rewards).squeeze(0).squeeze(1)

        if disable_mopo:
            mopo_lam = 0
            morel_thresh = 0
        else:
            mopo_lam = self.mopo_lam
            morel_thresh = self._morel_thresh

        # TODO: Handle 1 dimensional states
        if mopo_lam != 0 or morel_thresh:

            mopo_pen = self.get_penalty(self.params['mopo_penalty_type'], nextstates, nextstates_all, nextstates_logvar, rewards, rewards_all, rewards_logvar, nextstates_all_mu, rewards_all_mu, allocation)

            if self.use_automatic_lam_tuning:
                mopo_lam = self.log_mopo_lam.exp()
                # save this for now because this method is called under no_grad
                self._uncertainty_diff_vector = (mopo_lam * mopo_pen - self.target_uncertainty).detach()
            else:
                mopo_lam = self.mopo_lam

            if mopo_lam != 0:
                rewards = rewards - mopo_lam * mopo_pen
            elif self._morel_thresh:
                rewards[mopo_pen > self._morel_thresh] += self._morel_halt_reward

            if self.params['env_name'] == 'AntMOPOEnv':
                rewards += 1
        if IRL:
            if return_var:
                nextstates_var = self.get_total_variance(nextstates_all, nextstates_logvar)
                rewards_var = self.get_total_variance(rewards_all, rewards_logvar)
                return (nextstates, nextstates_var), (rewards, rewards_var), -mopo_lam*mopo_pen
            return nextstates, rewards, -mopo_lam*mopo_pen
        
        if return_var:
            nextstates_var = self.get_total_variance(nextstates_all, nextstates_logvar)
            rewards_var = self.get_total_variance(rewards_all, rewards_logvar)
            return (nextstates, nextstates_var), (rewards, rewards_var), mopo_pen
        else:
            return nextstates, rewards, mopo_pen

    def get_mopo_pen(self, state, action, get_var=True, deterministic=True, mopo_penalty_type=None):
        """Randomly allocate the data through the different dynamics models"""
        nextstates_all, rewards_all = self.get_next_states_rewards(state, action, get_var, deterministic)

        if get_var:
            nextstates_all, nextstates_logvar = nextstates_all.chunk(2, dim=2)
            rewards_all, rewards_logvar = rewards_all.chunk(2, dim=2)

        if get_var:
            nextstates_all, nextstates_logvar = nextstates_all.chunk(2, dim=2)
        if self.mopo_lam != 0:
            if mopo_penalty_type == 'mopo_default':
                nextstates_std = nextstates_logvar.exp().sqrt()
                mopo_pen = nextstates_std.norm(2, 2).amax(0)
            elif mopo_penalty_type == 'ensemble_var':
                nextstates_var = nextstates_logvar.exp()
                mean_of_vars = torch.mean(nextstates_var, dim=0)
                var_of_means = torch.var(nextstates_all, dim=0)
                print(mean_of_vars.shape)
                print(var_of_means.shape)
                vr = mean_of_vars + var_of_means
                mopo_pen = torch.mean(vr, dim=1)
            elif mopo_penalty_type == 'ensemble_std':
                nextstates_var = nextstates_logvar.exp()
                mean_of_vars = torch.mean(nextstates_var, dim=0)
                var_of_means = torch.var(nextstates_all, dim=0)
                std = (mean_of_vars + var_of_means).sqrt()
                mopo_pen = torch.mean(std, dim=1)
            elif self.params['mopo_penalty_type'] == 'ensemble_var_rew':
                rewards_var = rewards_logvar.exp()
                mean_of_vars = torch.mean(rewards_var, dim=0)
                var_of_means = torch.var(rewards_all, dim=0)
                vr = mean_of_vars + var_of_means
                mopo_pen = torch.mean(vr, dim=1)
            elif mopo_penalty_type == 'due' or mopo_penalty_type == 'sparse_gp':
                combined = torch.cat((state, action), dim=1)
                combined = torch.tensor_split(combined, 50)
                mopo_pens = []
                for c in combined:
                    mopo_pens.append(self.due_override.predict_var(c))
                mopo_pen = torch.cat(mopo_pens, dim=0)
                # mopo_pen = torch.mean(mopo_pen, dim=1)
            else:
                raise NotImplementedError

        return mopo_pen

    def update_lambda(self):
        lam_loss = (self.log_mopo_lam * self._uncertainty_diff_vector).mean()
        self.lam_optimizer.zero_grad()
        lam_loss.backward()
        self.lam_optimizer.step()

    @staticmethod
    def get_total_variance(mean_values, logvar_values):
        return (torch.var(mean_values, dim=0) + torch.mean(logvar_values.exp(), dim=0)).squeeze()

    def _get_validation_losses(self, validation_loader, get_weights=True):
        best_losses = []
        best_weights = []
        for model in self.models.values():
            best_losses.append(model.get_validation_loss(validation_loader))
            if get_weights:
                best_weights.append(deepcopy(model.state_dict()))
        best_losses = np.array(best_losses)
        return best_losses, best_weights

    def check_validation_losses(self, validation_loader):
        improved_any = False
        current_losses, current_weights = self._get_validation_losses(validation_loader, get_weights=True)
        improvements = ((self.current_best_losses - current_losses) / self.current_best_losses) > 0.01
        for i, improved in enumerate(improvements):
            if improved:
                self.current_best_losses[i] = current_losses[i]
                self.current_best_weights[i] = current_weights[i]
                improved_any = True
        return improved_any, current_losses

    def train_model(self, max_epochs: int = 100, n_samples: int = 200000, d4rl_init=False, save_model=False,
                    min_model_epochs=None):
        self.current_best_losses = np.zeros(
            self.params['num_models']) + sys.maxsize  # weird hack (YLTSI), there's almost surely a better way...
        self.current_best_weights = [None] * self.params['num_models']
        val_improve = deque(maxlen=6)
        lr_lower = False
        min_model_epochs = 0 if not min_model_epochs else min_model_epochs
        if d4rl_init:
            # Train on the full buffer until convergence, should be under 5k epochs
            n_samples = len(self.memory)
            n_samples_val = len(self.memory_val)
            max_epochs = 1000
        elif len(self.memory) < n_samples:
            n_samples = len(self.memory)
            n_samples_val = len(self.memory_val)
        else:
            n_samples_val = int(np.floor((n_samples / (1 - self.train_val_ratio)) * (self.train_val_ratio)))

        samples_train = self.memory.sample(n_samples)
        samples_validate = self.memory_val.sample(n_samples_val)

        # TODO: shift the training and val dataset using the fn 'get_max_state_action_uncertainty'

        batch_size = 256
        if n_samples_val == len(self.memory_val):
            samples_validate = self.memory_val.sample_all()
        else:
            samples_validate = self.memory_val.sample(n_samples_val)
        ########## MIX VALDIATION AND TRAINING ##########
        new_samples_train_dict = dict.fromkeys(samples_train._fields)
        new_samples_validate_dict = dict.fromkeys(samples_validate._fields)
        randperm = np.random.permutation(n_samples + n_samples_val)

        train_idx, valid_idx = randperm[:n_samples], randperm[n_samples:]
        assert len(valid_idx) == n_samples_val

        for i, key in enumerate(samples_train._fields):
            train_vals = samples_train[i]
            valid_vals = samples_validate[i]
            all_vals = np.array(list(train_vals) + list(valid_vals))
            train_vals = all_vals[train_idx]
            valid_vals = all_vals[valid_idx]
            new_samples_train_dict[key] = tuple(train_vals)
            new_samples_validate_dict[key] = tuple(valid_vals)

        samples_train = Transition(**new_samples_train_dict)
        samples_validate = Transition(**new_samples_validate_dict)
        ########## MIX VALDIATION AND TRAINING ##########
        transition_loader = DataLoader(
            EnsembleTransitionDataset(samples_train, self.state_filter, self.action_filter, n_models=self.num_models),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True
        )
        validate_dataset = TransitionDataset(samples_validate, self.state_filter, self.action_filter)
        sampler = SequentialSampler(validate_dataset)
        validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True
        )

        ### check validation before first training epoch
        improved_any, iter_best_loss = self.check_validation_losses(validation_loader)
        val_improve.append(improved_any)
        best_epoch = 0
        model_idx = 0
        print('Epoch: %s, Total Loss: N/A' % (0))
        print('Validation Losses:')
        print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
        for i in range(max_epochs):
            t0 = time.time()
            total_loss = 0
            loss = 0
            step = 0
            # value to shuffle dataloader rows by so each epoch each model sees different data
            perm = np.random.choice(self.num_models, size=self.num_models, replace=False)
            for x_batch, diff_batch, r_batch in transition_loader:
                x_batch = x_batch[:, perm]
                diff_batch = diff_batch[:, perm]
                r_batch = r_batch[:, perm]
                step += 1
                for idx in range(self.num_models):
                    loss += self.models[idx].train_model_forward(x_batch[:, idx], diff_batch[:, idx], r_batch[:, idx])
                total_loss = loss.item()
                if self.is_probabilistic:
                    loss += 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = 0
            t1 = time.time()
            print("Epoch training took {} seconds".format(t1 - t0))
            if (i + 1) % 1 == 0:
                improved_any, iter_best_loss = self.check_validation_losses(validation_loader)
                print('Epoch: {}, Total Loss: {}'.format(int(i + 1), float(total_loss)))
                print('Validation Losses:')
                print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
                print('Best Validation Losses So Far:')
                print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(self.current_best_losses)))
                val_improve.append(improved_any)
                if improved_any:
                    best_epoch = (i + 1)
                    print('Improvement detected this epoch.')
                else:
                    epoch_diff = i + 1 - best_epoch
                    plural = 's' if epoch_diff > 1 else ''
                    print('No improvement detected this epoch: {} Epoch{} since last improvement.'.format(epoch_diff,
                                                                                                          plural))
                if len(val_improve) > 5:
                    if not any(np.array(val_improve)[1:]):
                        # assert val_improve[0]
                        if (i >= min_model_epochs):
                            print('Validation loss stopped improving at %s epochs' % (best_epoch))
                            for model_index in self.models:
                                self.models[model_index].load_state_dict(self.current_best_weights[model_index])
                            self._select_elites(validation_loader)
                            if save_model:
                                self._save_model()
                            return
                        elif not lr_lower:
                            self._lr_scheduler.step()
                            lr_lower = True
                            val_improve = deque(maxlen=6)
                            val_improve.append(True)
                            print("Lowering Adam Learning for fine-tuning")
                t2 = time.time()
                print("Validation took {} seconds".format(t2 - t1))
        self._select_elites(validation_loader)
        if save_model:
            self._save_model()

    def set_model_logvar_limits(self):
        if isinstance(self.max_logvar, dict):
            for i, model in enumerate(self.models.values()):
                model.model.update_logvar_limits(self.max_logvar[self._model_groups[i]], self.min_logvar[self._model_groups[i]])
        else:
            for model in self.models.values():
                model.model.update_logvar_limits(self.max_logvar, self.min_logvar)

    def _select_elites(self, validation_loader):
        val_losses, _ = self._get_validation_losses(validation_loader, get_weights=False)
        print('Sorting Models from most to least accurate...')
        models_val_rank = val_losses.argsort()
        val_losses.sort()
        print('\nModel validation losses: {}'.format(val_losses))
        self.models = {i: self.models[idx] for i, idx in enumerate(models_val_rank)}
        self._elites_idx = list(range(self.num_elites))
        self.elites = {i: self.models[j] for i, j in enumerate(self._elites_idx)}
        self.elite_errors = {i: val_losses[j] for i, j in enumerate(self._elites_idx)}
        print('\nSelected the following models as elites: {}'.format(self._elites_idx))
        return val_losses

    def _save_model(self):
        """
        Method to save model after training is completed
        """
        print("Saving model checkpoint...")
        check_or_make_folder("./checkpoints")
        check_or_make_folder("./checkpoints/model_saved_weights")
        save_dir = "./checkpoints/model_saved_weights/{}".format(self._model_id)
        check_or_make_folder(save_dir)
        # Create a dictionary with pytorch objects we need to save, starting with models
        torch_state_dict = {'model_{}_state_dict'.format(i): w for i, w in enumerate(self.current_best_weights)}
        # Then add logvariance limit terms
        torch_state_dict['logvar_min'] = self.min_logvar
        torch_state_dict['logvar_max'] = self.max_logvar
        # Save Torch files
        torch.save(torch_state_dict, save_dir + "/torch_model_weights.pt")
        # Create a dict containing training and validation datasets
        data_state_dict = {'train_buffer': self.memory, 'valid_buffer': self.memory_val,
                           'state_filter': self.state_filter, 'action_filter': self.action_filter}
        # Then add validation performance for checking purposes during loading (i.e., make sure we got the same performance)
        data_state_dict['validation_performance'] = self.current_best_losses
        # Pickle the data dict
        pickle.dump(data_state_dict, open(save_dir + '/model_data.pkl', 'wb'))
        print("Saved model snapshot trained on {} datapoints".format(len(self.memory)))

    def load_model(self, model_dir):
        """
        Method to load model from checkpoint folder
        """
        # Check that the environment matches the dir name
        assert self._env_name.split('-')[
                   0].lower() in model_dir.lower(), "Model loaded was not trained on this environment"

        print("Loading model from checkpoint...")

        torch_state_dict = torch.load(model_dir + '/torch_model_weights.pt', map_location=device)
        for i in range(self.num_models):
            self.models[i].load_state_dict(torch_state_dict['model_{}_state_dict'.format(i)])
        self.min_logvar = torch_state_dict['logvar_min']
        self.max_logvar = torch_state_dict['logvar_max']

        data_state_dict = pickle.load(open(model_dir + '/model_data.pkl', 'rb'))
        # Backwards Compatability
        if isinstance(data_state_dict['train_buffer'], ReplayPool):
            assert self.memory.capacity > len(data_state_dict['train_buffer'])
            assert self.memory_val.capacity > len(data_state_dict['valid_buffer'])
            all_train = data_state_dict['train_buffer'].sample_all()._asdict()
            all_train = Transition(**{k: np.stack(v) for k, v in all_train.items()})
            all_valid = data_state_dict['valid_buffer'].sample_all()._asdict()
            all_valid = Transition(**{k: np.stack(v) for k, v in all_valid.items()})
            self.memory.push(all_train)
            self.memory_val.push(all_valid)
        else:
            self.memory, self.memory_val = data_state_dict['train_buffer'], data_state_dict['valid_buffer']
        self.state_filter, self.action_filter = data_state_dict['state_filter'], data_state_dict['action_filter']

        # Confirm that we retrieve the checkpointed validation performance
        all_valid = self.memory_val.sample_all()
        validate_dataset = TransitionDataset(all_valid, self.state_filter, self.action_filter)
        sampler = SequentialSampler(validate_dataset)
        validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=256,
            pin_memory=True
        )

        val_losses = self._select_elites(validation_loader)
        self.set_model_logvar_limits()

        model_id = model_dir.split('/')[-1]
        self._model_id = model_id

        return val_losses

        # Doesn't work because we mix up val and train data
        # assert np.isclose(val_losses, data_state_dict['validation_performance']).all()

    def load_model_from_population(self, model_dirs):
        """
        Method to load model from aggregated population
        """

        # Aggregate the state dictionaries first and then relabel them
        aggregate_torch_state_dict = {}
        aggregated_logvar_min = {}
        aggregated_logvar_max = {}
        self._model_groups = {}
        for model_idx, model_dir in enumerate(model_dirs):
            # Check that the environment matches the dir name
            assert self._env_name.split('-')[
                       0].lower() in model_dir.lower(), "Model loaded was not trained on this environment"
            torch_state_dict = torch.load(model_dir + '/torch_model_weights.pt', map_location=device)
            for key, value in torch_state_dict.items():
                if 'model' in key:
                    cur_idx = int(key.split('_')[1])
                    self._model_groups[model_idx * 7 + cur_idx] = model_idx
                    relabelled_key = 'model_{}_state_dict'.format(model_idx * 7 + cur_idx)
                    aggregate_torch_state_dict[relabelled_key] = value
                else:
                    if key == 'logvar_min':
                        aggregated_logvar_min[model_idx] = value
                    elif key == 'logvar_max':
                        aggregated_logvar_max[model_idx] = value

        for i in range(self.num_models):
            self.models[i].load_state_dict(aggregate_torch_state_dict['model_{}_state_dict'.format(i)])
        self.min_logvar = aggregated_logvar_min
        self.max_logvar = aggregated_logvar_max

        # Takes the data from the first element of the population, shouldn't matter?
        data_state_dict = pickle.load(open(model_dirs[0] + '/model_data.pkl', 'rb'))
        # Backwards Compatability
        if isinstance(data_state_dict['train_buffer'], ReplayPool):
            assert self.memory.capacity > len(data_state_dict['train_buffer'])
            assert self.memory_val.capacity > len(data_state_dict['valid_buffer'])
            all_train = data_state_dict['train_buffer'].sample_all()._asdict()
            all_train = Transition(**{k: np.stack(v) for k, v in all_train.items()})
            all_valid = data_state_dict['valid_buffer'].sample_all()._asdict()
            all_valid = Transition(**{k: np.stack(v) for k, v in all_valid.items()})
            self.memory.push(all_train)
            self.memory_val.push(all_valid)
        else:
            self.memory, self.memory_val = data_state_dict['train_buffer'], data_state_dict['valid_buffer']
        self.state_filter, self.action_filter = data_state_dict['state_filter'], data_state_dict['action_filter']

        # Confirm that we retrieve the checkpointed validation performance
        all_valid = self.memory_val.sample_all()
        validate_dataset = TransitionDataset(all_valid, self.state_filter, self.action_filter)
        sampler = SequentialSampler(validate_dataset)
        validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=1024,
            pin_memory=True
        )

        self.set_model_logvar_limits()
        val_losses = self._select_elites(validation_loader)

        model_id = model_dirs[0].split('/')[-1]
        self._model_id = model_id

        return val_losses

    def remove_model(self):
        """ removes the least accurate model """

        if self.num_models - 1 < self.num_elites:
            print("Can't remove any more models, otherwise we'll start removing elites")
            return

        self.num_models -= 1
        self._removed_models.append(self.models[self.num_models])
        del self.models[self.num_models]
        print("Removed a model; you have {} models".format(self.num_models))

    def add_models(self):
        """ adds back the most accurate removed model """

        if len(self._removed_models) < 1:
            print("You haven't removed any models!")
            return

        model = self._removed_models.pop()
        self.models[self.num_models] = model
        self.num_models += 1
        print("Re-added a model; you have {} models".format(self.num_models))

    def get_replay_buffer_predictions(self, only_validation=False, return_sample=False):
        """ Gets the predictions of all ensemble members on the data currently in the buffer """
        buffer_data = self.memory_val.sample_all()
        if not only_validation:
            pass
        dataset = TransitionDataset(buffer_data, self.state_filter, self.action_filter)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1024,
            pin_memory=True
        )

        preds = torch.stack([m.get_predictions_from_loader(dataloader, return_sample=return_sample) for m in self.models.values()], 0)

        return preds

class Model(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 1024,
                 is_probabilistic=True,
                 is_done_func=None,
                 reward_head=1,
                 seed=0,
                 l2_reg_multiplier=1.,
                 num=0):

        super(Model, self).__init__()
        torch.manual_seed(seed)
        if is_probabilistic:
            self.model = BayesianNeuralNetwork(input_dim, output_dim, 200, is_done_func, reward_head, l2_reg_multiplier,
                                               seed)
        else:
            self.model = VanillaNeuralNetwork(input_dim, output_dim, h, is_done_func, reward_head, seed)
        self.is_probabilistic = self.model.is_probabilistic
        self.weights = self.model.weights
        self.reward_head = reward_head

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def get_next_state_reward(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter,
                              keep_logvar=False, deterministic=False, return_mean=False):
        return self.model.get_next_state_reward(state, action, state_filter, action_filter, keep_logvar,
                                                deterministic, return_mean)

    def predict_state(self, state: np.array, action: np.array, state_filter, action_filter) -> (np.array, float):
        state, action = torch.Tensor(state).to(device), torch.Tensor(action).to(device)
        nextstate, reward = self.get_next_state_reward(state, action, state_filter, action_filter)
        nextstate = nextstate.detach().cpu().numpy()
        if self.reward_head:
            reward = reward.detach().cpu().item()
        return nextstate, reward

    def get_state_action_uncertainty(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter):
        return self.model.get_state_action_uncertainty(state, action, state_filter, action_filter)

    def _train_model_forward(self, x_batch):
        self.model.train()
        self.model.zero_grad()
        x_batch = x_batch.to(device, non_blocking=True)
        y_pred = self.forward(x_batch)
        return y_pred

    def train_model_forward(self, x_batch, delta_batch, r_batch):
        delta_batch, r_batch = delta_batch.to(device, non_blocking=True), r_batch.to(device, non_blocking=True)
        y_pred = self._train_model_forward(x_batch)
        y_batch = torch.cat([delta_batch, r_batch.unsqueeze(dim=1)], dim=1) if self.reward_head else delta_batch
        loss = self.model.loss(y_pred, y_batch)
        return loss

    def get_predictions_from_loader(self, data_loader, return_targets = False, return_sample=False):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_batch_val, delta_batch_val, r_batch_val in data_loader:
                x_batch_val, delta_batch_val, r_batch_val = x_batch_val.to(device,
                                                                           non_blocking=True), delta_batch_val.to(
                    device, non_blocking=True), r_batch_val.to(device, non_blocking=True)
                y_pred_val = self.forward(x_batch_val)
                preds.append(y_pred_val)
                if return_targets:
                    y_batch_val = torch.cat([delta_batch_val, r_batch_val.unsqueeze(dim=1)],
                        dim=1) if self.reward_head else delta_batch_val
                    targets.append(y_batch_val)
        
        preds = torch.vstack(preds)

        if return_sample:
            mu, logvar = preds.chunk(2, dim=1)
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            sample = dist.sample()
            preds = torch.cat((sample, preds), dim=1)

        if return_targets:
            targets = torch.vstack(targets)
            return preds, targets
        else:
            return preds
                
    def get_validation_loss(self, validation_loader):
        self.model.eval()
        preds, targets = self.get_predictions_from_loader(validation_loader, return_targets=True)
        if self.is_probabilistic:
            return self.model.loss(preds, targets, logvar_loss=False).item()
        else:
            return self.model.loss(preds, targets).item()

    def get_acquisition(self, rollout: List[Transition], state_filter, action_filter):
        self.model.eval()
        state = []
        action = []
        nextstate = []
        reward = []
        # for rollout in rollouts:
        for step in rollout:
            state.append(step.state)
            action.append(step.action)
            nextstate.append(step.nextstate)
            reward.append(step.reward)
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward).reshape(-1, 1)
        state_action_filtered, delta_filtered = prepare_data(
            state,
            action,
            nextstate,
            state_filter,
            action_filter)
        y_true = np.concatenate((delta_filtered, reward), axis=1) if self.reward_head else delta_filtered
        y_true, state_action_filtered = torch.Tensor(y_true).to(device), torch.Tensor(state_action_filtered).to(device)
        y_pred = self.forward(state_action_filtered)

        if self.is_probabilistic:
            loss = self.model.loss(y_pred, y_true, logvar_loss=False)
        else:
            loss = self.model.loss(y_pred, y_true)

        return float(loss.item())


class VanillaNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 1024,
                 is_done_func=None,
                 reward_head=True,
                 seed=0):

        super().__init__()
        torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU()
        )
        self.delta = nn.Linear(h, output_dim)
        params = list(self.network.parameters()) + list(self.delta.parameters())
        self.weights = params
        self.to(device)
        self.loss = nn.MSELoss()
        self.is_done_func = is_done_func
        self.reward_head = reward_head

    @property
    def is_probabilistic(self):
        return False

    def forward(self, x: torch.Tensor):
        hidden = self.network(x)
        delta = self.delta(hidden)
        return delta

    @staticmethod
    def filter_inputs(state, action, state_filter, action_filter):
        state_f = state_filter.filter_torch(state)
        action_f = action_filter.filter_torch(action)
        state_action_f = torch.cat((state_f, action_f), dim=1)
        return state_action_f

    def get_next_state_reward(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter,
                              keep_logvar=False):
        if keep_logvar:
            raise Exception("This is a deterministic network, there is no logvariance prediction")
        state_action_f = self.filter_inputs(state, action, state_filter, action_filter)
        y = self.forward(state_action_f)
        if self.reward_head:
            diff_f = y[:, :-1]
            reward = y[:, -1].unsqueeze(1)
        else:
            diff_f = y
            reward = 0
        diff = diff_f
        nextstate = state + diff
        return nextstate, reward

    def get_state_action_uncertainty(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter):
        raise Exception("This is a deterministic network, there is no logvariance prediction")


class BayesianNeuralNetwork(VanillaNeuralNetwork):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 200,
                 is_done_func=None,
                 reward_head=True,
                 l2_reg_multiplier=1.,
                 seed=0):
        super().__init__(input_dim,
                         output_dim,
                         h,
                         is_done_func,
                         reward_head,
                         seed)
        torch.manual_seed(seed)
        del self.network
        self.fc1 = nn.Linear(input_dim, h)
        reinitialize_fc_layer_(self.fc1)
        self.fc2 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc2)
        self.fc3 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc3)
        self.fc4 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc4)
        self.use_blr = False
        self.delta = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.delta)
        self.logvar = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.logvar)
        self.loss = GaussianMSELoss()
        self.activation = nn.SiLU()
        self.lambda_prec = 1.0
        self.max_logvar = None
        self.min_logvar = None
        params = []
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.delta, self.logvar]
        self.decays = np.array([0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]) * l2_reg_multiplier
        for layer, decay in zip(self.layers, self.decays):
            params.extend(get_weight_bias_parameters_with_decays(layer, decay))
        self.weights = params
        self.to(device)

    def get_l2_reg_loss(self):
        l2_loss = 0
        for layer, decay in zip(self.layers, self.decays):
            for name, parameter in layer.named_parameters():
                if 'weight' in name:
                    l2_loss += parameter.pow(2).sum() / 2 * decay
        return l2_loss

    def update_logvar_limits(self, max_logvar, min_logvar):
        self.max_logvar, self.min_logvar = max_logvar, min_logvar

    @property
    def is_probabilistic(self):
        return True

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        delta = self.delta(x)
        logvar = self.logvar(x)
        # Taken from the PETS code to stabilise training
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return torch.cat((delta, logvar), dim=1)

    def get_next_state_reward(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter,
                              keep_logvar=False, deterministic=False, return_mean=False):
        state_action_f = self.filter_inputs(state, action, state_filter, action_filter)
        mu, logvar = self.forward(state_action_f).chunk(2, dim=1)
        mu_orig = mu
        if not deterministic:
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            mu = dist.sample()
        if self.reward_head:
            mu_diff_f = mu[:, :-1]
            logvar_diff_f = logvar[:, :-1]
            mu_reward = mu[:, -1].unsqueeze(1)
            logvar_reward = logvar[:, -1].unsqueeze(1)
            mu_diff_f_orig = mu_orig[:, :-1]
            mu_reward_orig = mu_orig[:, -1].unsqueeze(1)
        else:
            mu_diff_f = mu
            logvar_diff_f = logvar
            mu_reward = torch.zeros_like(mu[:, -1].unsqueeze(1))
            logvar_reward = torch.zeros_like(logvar[:, -1].unsqueeze(1))
            mu_diff_f_orig = mu_orig
            mu_reward_orig = mu_reward
        mu_diff = mu_diff_f
        mu_nextstate = state + mu_diff
        logvar_nextstate = logvar_diff_f
        if return_mean:
            mu_nextstate = torch.cat((mu_nextstate, mu_diff_f_orig + state), dim=1)
            mu_reward = torch.cat((mu_reward, mu_reward_orig), dim=1)
        if keep_logvar:
            return (mu_nextstate, logvar_nextstate), (mu_reward, logvar_reward)
        else:
            return mu_nextstate, mu_reward

    def get_state_action_uncertainty(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter):
        state_action_f = self.filter_inputs(state, action, state_filter, action_filter)
        _, logvar = self.forward(state_action_f).chunk(2, dim=1)
        # TODO: See above, which is the correct logvar?
        return logvar


def reinitialize_fc_layer_(fc_layer):
    """
    Helper function to initialize a fc layer to have a truncated normal over the weights, and zero over the biases
    """
    input_dim = fc_layer.weight.shape[1]
    std = get_trunc_normal_std(input_dim)
    torch.nn.init.trunc_normal_(fc_layer.weight, std=std, a=-2 * std, b=2 * std)
    torch.nn.init.zeros_(fc_layer.bias)


def get_trunc_normal_std(input_dim):
    """
    Returns the truncated normal standard deviation required for weight initialization
    """
    return 1 / (2 * np.sqrt(input_dim))


def get_weight_bias_parameters_with_decays(fc_layer, decay):
    """
    For the fc_layer, extract only the weight from the .parameters() method so we don't regularize the bias terms
    """
    decay_params = []
    non_decay_params = []
    for name, parameter in fc_layer.named_parameters():
        if 'weight' in name:
            decay_params.append(parameter)
        elif 'bias' in name:
            non_decay_params.append(parameter)

    decay_dicts = [{'params': decay_params, 'weight_decay': decay}, {'params': non_decay_params, 'weight_decay': 0.}]

    return decay_dicts
