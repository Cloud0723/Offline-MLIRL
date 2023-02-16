from collections import deque
import random
import pdb
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EnsembleGymEnv
from sac import SAC_Agent
from utils import (filter_torch, filter_torch_invert, get_residual, get_stats,
                   random_env_forward, torch_reward, Transition, TransitionContext,ReplayPool, check_or_make_folder)
from utils import ReplayPoolCtxt, ReplayPool, FasterReplayPool, FasterReplayPoolCtxt, TanhTransform, Transition, TransitionContext, filter_torch
import d4rl
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class reward_estimator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes=(256,256),
        hid_act='relu',
        use_bn=False,
        residual=False,
        clamp_magnitude=10.0,
        device=device,
        **kwargs
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim
        self.device = device
        self.residual = residual

        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_bn: block.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        output = self.last_fc(x)
        output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return output  

    def r(self, batch):
        return self.forward(batch)

    def get_scalar_reward(self, obs):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            reward = self.forward(obs).cpu().detach().numpy().flatten()
        self.train()
        return reward
    

class Trainer(object):

    def __init__(self, params, model: EnsembleGymEnv, agent: SAC_Agent, device):
        self.agent = agent
        self.model = model
        
        self.expert_pool = ReplayPool(capacity=1e6)
        
        self.envname=params['env_name'].split('-')[0]
        self.load_expert_dataset()
        
        if params['env_name'].split('-')[0]=='hopper':
            self.reward_function=reward_estimator(14).to(device)
        else:
            self.reward_function=reward_estimator(23).to(device)
        self.transfer=params['transfer']
        self.reward_opti= torch.optim.Adam(self.reward_function.parameters(), lr=1e-4,weight_decay=1e-3, betas=(0.9, 0.999))
        if self.transfer:
            #self.reward_function.load_state_dict(torch.load('/home/luoqijun/code/IRL_Code/MBIRL/rethinking-code-supp/medexp_halfcheetah_reward_780.pt'))
            #self.reward_function.load_state_dict(torch.load('/home/luoqijun/code/IRL_Code/MBIRL/rethinking-code-supp/reward_hopper/medexp_hopper_reward_2602_.pt'))
            self.reward_function.load_state_dict(torch.load('/home/luoqijun/code/IRL_Code/MBIRL/rethinking-code-supp/reward_walker/medexp_walker_reward_5603_.pt'))
            
            self.reward_function.eval()
        self._init_collect = params['init_collect']
        self._max_model_epochs = params['model_epochs']
        self._var_type = params['var_type']
        self._num_rollouts = params['num_rollouts_per_step'] * params['model_train_freq']
        self._model_retain_epochs = params['model_retain_epochs']
        self._device = device
        self._train_policy_every = params['train_policy_every']
        self._reward_head = params['reward_head']
        self._policy_update_steps = params['policy_update_steps']
        self._steps_k = params['steps_k']
        self._reward_step = params['reward_steps']
        if isinstance(self._steps_k, list):
            self._cur_steps_k = self._steps_k[0]
        else:
            self._cur_steps_k = self._steps_k
        self._n_eval_rollouts = params['n_eval_rollouts']
        self._real_sample_ratio = params['real_sample_ratio']
        self._model_train_freq = params['model_train_freq']
        self._rollout_model_freq = params['rollout_model_freq']
        self._oac = params['oac']
        self._sample_states = params['states']
        self._done = True
        self._state = None
        self._n_epochs = 0
        self._is_done_func = params['is_done_func']
        self._var_thresh = params['var_thresh']
        self._keep_logvar = True if self._var_thresh is not None else False
        self.k_used = [0]
        self._espi = params['espi']
        self._max_steps = params['epoch_steps'] if params[
            'epoch_steps'] else self.model.real_env.env.spec.max_episode_steps
        self._env_step = 0
        self._curr_rollout = []
        self._deterministic = params['deterministic_rollouts']
        self._seed = params['seed']
        self._min_model_epochs = params['min_model_epochs']
        if self._min_model_epochs:
            assert self._min_model_epochs < self._max_model_epochs, "Can't have a min epochs that is less than the max"
        self._augment_offline_data = params['augment_offline_data']

        if params['population_model_dirs']:
            self._load_population_models(params)

        self._morel_halt_reward = params['morel_halt_reward']

        # Remove in a sec, just for testing
        self._params = params

        if self._params['mopo'] and self._params['morel']:
            raise Exception('Do not use MOReL and MOPO together please')

    def load_expert_dataset(self):
        import os
        current_path=os.path.dirname(os.path.abspath(__file__))+'/expert_data/'+self.envname+'/'
        state=np.load(current_path+'states.npy')[:50]
        action=np.load(current_path+'actions.npy')[:50]
        dones=np.load(current_path+'dones.npy')[:50]

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                self.expert_pool.push(Transition(state[i][j], action[i][j], 0 , 0, dones[i][j]))
    
    def _train_model(self, d4rl_init=False, save_model=False):
        print("\nTraining Model...")
        self.model.train_model(self._max_model_epochs, d4rl_init=d4rl_init, save_model=save_model,
                               min_model_epochs=self._min_model_epochs)


    #TODO: We have written this code like 3 times now, maybe incorporate it with myopic thing
    def rollout_with_ground_truth(self, policy: SAC_Agent, num_parallel = 100, num_steps=25):
        """
        Rolls out 'policy' in the WM for 'num_steps' with 'num_parallel' starting points
        Also returns the uncertainty penalties of interest along the way
        Also returns the MSE w.r.t. ground truth for our true D_TV 
        """
        start_states = torch.FloatTensor(
                np.array(self.model.model.memory.sample(num_parallel, unique=False)[0])).to(self._device)
        done_false = [False for _ in range(start_states.shape[0])]
        t = 0
        state = start_states
        idxs_remaining = np.arange(start_states.shape[0])
        stats_dic = {
            'groundtruth_mses': np.zeros((num_parallel, num_steps)),
            'mopo_paper': np.zeros((num_parallel, num_steps)),
            'morel': np.zeros((num_parallel, num_steps)),
            'lompo': np.zeros((num_parallel, num_steps)),
            'm2ac': np.zeros((num_parallel, num_steps)),
            'ensemble_var': np.zeros((num_parallel, num_steps)),
            'ensemble_std': np.zeros((num_parallel, num_steps))
        }
        while t < num_steps:
            t += 1
            with torch.no_grad():
                # Get deterministic action
                _, _, action_det = policy.policy(state)
                penalties, nextstate = self.model.model.get_all_penalties_state_action(state, action_det)
                true_nextstate = self._get_ground_truth_nextstate(state, action_det)
                mses = ((nextstate.cpu().numpy() - true_nextstate)**2).mean(1)
                if self._is_done_func:
                    done = self._is_done_func(nextstate).cpu().numpy().tolist()
                else:
                    done = done_false[:nextstate.shape[0]]
                not_done = ~np.array(done)
                for k,v in penalties.items():
                    stats_dic[k][idxs_remaining, t-1] = v
                stats_dic['groundtruth_mses'][idxs_remaining, t-1] = mses
                idxs_remaining = idxs_remaining[not_done]
                if len(nextstate.shape) == 1:
                    nextstate.unsqueeze_(0)
                state = nextstate[not_done]
                if len(state.shape) == 1:
                    state.unsqueeze_(0)
                if not_done.sum() == 0:
                    break
        return stats_dic
    
    def _get_ground_truth_nextstate(self, state, action):
        env = self.model.eval_env
        env.reset()
        state = state.cpu().numpy()
        action = action.cpu().numpy()
        qpos0 = np.array([0])
        ground_truth_ns = []
        if 'hopper' in env.spec.id.lower():
            for s, a in zip(state, action):
                qpos = np.concatenate([qpos0, s[:5]])
                qvel = s[5:]
                env.set_state(qpos, qvel)
                true_ns = env.step(a)
                ground_truth_ns.append(true_ns[0])
        else:
            raise NotImplementedError
        return np.stack(ground_truth_ns)
    
    def _rollout_model_expert(self,start_states,sample_pool=None,IRL=True,Penalty_only=False):
        _num_rollouts=start_states.shape[0]
        
        self.model.convert_filter_to_torch()
        self.k_used = [0]
        self.var_mean = []
        
        done_true = [True for _ in range(_num_rollouts)]
        done_false = [False for _ in range(_num_rollouts)]
        
        state = start_states.clone()
        prev_done = done_false
        var = 0
        t = 0
        transition_count = 0
        
        while t < self._reward_step:
            t += 1
            if self._params['rad_rollout']:
                high = 1.2
                low = 0.8
                scale = high - low
                random_amplitude_scaling = (torch.rand(start_states.shape) * scale + low).to(device)

            with torch.no_grad():
                # TODO: Random steps intially?
                if self.agent.augment_sac and self._params['rad_rollout']:
                    # Normalise this here
                    # filter_torch(random_amplitude_scaling, 1, 0.11547)
                    augmented_state = torch.cat((state, random_amplitude_scaling), 1)
                    action, _, _ = self.agent.policy(augmented_state)
                elif self.agent.augment_sac and self._params['context_type'] == 'rad_augmentation':
                    # Assume a base context of 1s just as a filler.
                    augmented_state = torch.cat((state, torch.ones_like(state)), 1)
                    action, _, _ = self.agent.policy(augmented_state)
                elif self.agent.augment_sac and self._params['context_type'] == 'rad_magnitude':
                    # Assume a base context of 1s just as a filler.
                    augmented_state = torch.cat((state, torch.ones(state.shape[0], 1).to(device)), 1)
                    action, _, _ = self.agent.policy(augmented_state)
                else:
                    action, _, _ = self.agent.policy(state)

                nextstate, reward, penalties = self.model.model.random_env_step(state,
                                                                     action,
                                                                     get_var=self._keep_logvar,
                                                                     deterministic=self._deterministic,
                                                                     IRL=Penalty_only,
                                                                     )

                if self._keep_logvar:
                    nextstate, nextstate_var, reward, reward_var = nextstate[0], nextstate[1], reward[0], reward[1]

                nextstate_copy = nextstate.clone()
                if self._params['rad_rollout']:
                    nextstate *= random_amplitude_scaling
                    
            if self._is_done_func:
                done = self._is_done_func(nextstate).cpu().numpy().tolist()
            else:
                done = done_false[:nextstate.shape[0]]
            if self._params['morel'] and not self._params['morel_non_stop']:
                # we're going to go to the absorbing HALT state here
                # TODO: fix this to be more efficient?
                done = torch.tensor(done).to(device) | (penalties > self._morel_threshold)
                done = done.cpu().numpy().tolist()

            not_done = ~np.array(done)
            if self._keep_logvar:
                print("Reward Var: mean = {}, max = {}, min = {}".format(np.mean(reward_var.cpu().numpy()),
                                                                         np.max(reward_var.cpu().numpy()),
                                                                         np.min(reward_var.cpu().numpy())))
                self.var_mean.append(np.mean(reward_var.cpu().numpy()))
                var_low = reward_var < self._var_thresh
                done_k = np.array(done) + ~var_low.cpu().numpy().squeeze()
                not_done = not_done & var_low.cpu().numpy().squeeze()
                self.k_used += [t for _ in done_k if _ is True]
                
            uncert = 0
            if self._reward_head:
                reward = reward.cpu().detach().numpy()
            else:
                reward = torch_reward(self.model.name, nextstate, action, done)
            state_np, action_np, nextstate_np ,penalties= state.detach().cpu().numpy(), action.detach().cpu().numpy(), nextstate.detach().cpu().numpy(), penalties.detach().cpu().numpy()
            
            if self._params['rad_rollout'] and self.agent.augment_sac:
                rad_np = random_amplitude_scaling.detach().cpu().numpy()
                # for s, a, r, s_n, d, ctxt in zip(state_np, action_np, reward, nextstate_np, done, rad_np):
                #     r = r.item()
                #     self.agent.replay_pool.push(TransitionContext(s, a, r, s_n, d, ctxt))
                sample_pool.push(TransitionContext(state_np, action_np, penalties, nextstate_np, np.array(done), rad_np))
                
            else:
                # for s, a, r, s_n, d in zip(state_np, action_np, reward, nextstate_np, done):
                #     r = r.item()
                #     self.agent.replay_pool.push(Transition(s, a, r, s_n, d))
                sample_pool.push(Transition(state_np, action_np, penalties, nextstate_np, np.array(done)))
                
            if not_done.sum() == 0:
                print("Finished rollouts early: all terminated after %s timesteps" % (t))
                break
            transition_count += len(nextstate)
            # Initialize state clean to be augmented next step.
            if len(nextstate_copy.shape) == 1:
                nextstate_copy.unsqueeze_(0)
            state = nextstate_copy[not_done]
            if len(state.shape) == 1:
                state.unsqueeze_(0)
            print("Remaining = {}".format(np.round(state.shape[0]) / start_states.shape[0], 2))
            var += uncert ** 2
        print("Finished rollouts: all terminated after %s timesteps" % (t))
        print("Added {} transitions to agent replay pool".format(transition_count))
        
        return 
    
    
    def _rollout_model(self,sample_pool=None,IRL=False,Penalty_only=False):
        print("\nRolling out Policy in Model...")
        if self._params['mopo']:
            print("\nUsing MOPO Penalty")
        elif self._params['morel']:
            print("\nUsing MOReL Penalty")
        if self._var_type == 'reward':
            state_dynamics = False
        elif self._var_type == 'state':
            state_dynamics = True
        else:
            raise Exception("Variance must either be 'reward' or 'state'")

        for model in self.model.model.models.values():
            model.to(self._device)

        self.model.convert_filter_to_torch()
        self.k_used = [0]
        self.var_mean = []

        done_true = [True for _ in range(self._num_rollouts)]
        done_false = [False for _ in range(self._num_rollouts)]

        if self._sample_states == 'uniform':
            start_states = torch.FloatTensor(
                np.array(self.model.model.memory.sample(self._num_rollouts, unique=False)[0])).to(self._device)
        elif self._sample_states == 'entropy':
            all_states = [transition[0] for transition in self.model.model.memory.get_all()]
            all_states = torch.FloatTensor(all_states).to(self._device)
            _, _, all_actions = self.agent.policy(all_states)
            u = get_stats(self.model, all_states, all_actions, self.model.model.state_filter,
                          self.model.model.action_filter, False,
                          False, self._reward_head)
            scaled_neg_ent = -np.log(u)
            dist = softmax(scaled_neg_ent)
            sample = np.random.choice(all_states.shape[0], self._num_rollouts, p=dist.flatten())
            start_states = all_states[sample]

        state = start_states.clone()
        prev_done = done_false
        var = 0
        t = 0
        transition_count = 0
        
        while t < self._cur_steps_k:
            t += 1
            if self._params['rad_rollout']:
                high = 1.2
                low = 0.8
                scale = high - low
                random_amplitude_scaling = (torch.rand(start_states.shape) * scale + low).to(device)
                # Just scale nextstate for now
                # state *= random_amplitude_scaling

            with torch.no_grad():
                # TODO: Random steps intially?
                if self.agent.augment_sac and self._params['rad_rollout']:
                    # Normalise this here
                    # filter_torch(random_amplitude_scaling, 1, 0.11547)
                    augmented_state = torch.cat((state, random_amplitude_scaling), 1)
                    action, _, _ = self.agent.policy(augmented_state)
                elif self.agent.augment_sac and self._params['context_type'] == 'rad_augmentation':
                    # Assume a base context of 1s just as a filler.
                    augmented_state = torch.cat((state, torch.ones_like(state)), 1)
                    action, _, _ = self.agent.policy(augmented_state)
                elif self.agent.augment_sac and self._params['context_type'] == 'rad_magnitude':
                    # Assume a base context of 1s just as a filler.
                    augmented_state = torch.cat((state, torch.ones(state.shape[0], 1).to(device)), 1)
                    action, _, _ = self.agent.policy(augmented_state)
                else:
                    action, _, _ = self.agent.policy(state)
                # to do

                nextstate, reward, penalties = self.model.model.random_env_step(state,
                                                                     action,
                                                                     get_var=self._keep_logvar,
                                                                     deterministic=self._deterministic,
                                                                     IRL=Penalty_only,
                                                                     )

                if self._keep_logvar:
                    nextstate, nextstate_var, reward, reward_var = nextstate[0], nextstate[1], reward[0], reward[1]

                nextstate_copy = nextstate.clone()
                if self._params['rad_rollout']:
                    nextstate *= random_amplitude_scaling

            if self._params['tune_mopo_lam']:
                self.model.model.update_lambda()

            if self._is_done_func:
                done = self._is_done_func(nextstate).cpu().numpy().tolist()
            else:
                done = done_false[:nextstate.shape[0]]
            if self._params['morel'] and not self._params['morel_non_stop']:
                # we're going to go to the absorbing HALT state here
                # TODO: fix this to be more efficient?
                done = torch.tensor(done).to(device) | (penalties > self._morel_threshold)
                done = done.cpu().numpy().tolist()

            not_done = ~np.array(done)
            if self._keep_logvar:
                print("Reward Var: mean = {}, max = {}, min = {}".format(np.mean(reward_var.cpu().numpy()),
                                                                         np.max(reward_var.cpu().numpy()),
                                                                         np.min(reward_var.cpu().numpy())))
                self.var_mean.append(np.mean(reward_var.cpu().numpy()))
                var_low = reward_var < self._var_thresh
                done_k = np.array(done) + ~var_low.cpu().numpy().squeeze()
                not_done = not_done & var_low.cpu().numpy().squeeze()
                self.k_used += [t for _ in done_k if _ is True]
            uncert = 0
            if self._reward_head:
                reward = reward.cpu().detach().numpy()
            else:
                reward = torch_reward(self.model.name, nextstate, action, done)
            state_np, action_np, nextstate_np ,penalties= state.detach().cpu().numpy(), action.detach().cpu().numpy(), nextstate.detach().cpu().numpy(), penalties.detach().cpu().numpy()
            if self._params['rad_rollout'] and self.agent.augment_sac:
                rad_np = random_amplitude_scaling.detach().cpu().numpy()
                # for s, a, r, s_n, d, ctxt in zip(state_np, action_np, reward, nextstate_np, done, rad_np):
                #     r = r.item()
                #     self.agent.replay_pool.push(TransitionContext(s, a, r, s_n, d, ctxt))
                if IRL:
                    sample_pool.push(TransitionContext(state_np, action_np, penalties, nextstate_np, np.array(done), rad_np))
                else:
                    self.agent.replay_pool.push(TransitionContext(state_np, action_np, penalties, nextstate_np, np.array(done), rad_np))
            else:
                # for s, a, r, s_n, d in zip(state_np, action_np, reward, nextstate_np, done):
                #     r = r.item()
                #     self.agent.replay_pool.push(Transition(s, a, r, s_n, d))
                if IRL:
                    sample_pool.push(Transition(state_np, action_np, penalties, nextstate_np, np.array(done)))
                else:
                    self.agent.replay_pool.push(Transition(state_np, action_np, penalties, nextstate_np, np.array(done)))
            if not_done.sum() == 0:
                print("Finished rollouts early: all terminated after %s timesteps" % (t))
                break
            transition_count += len(nextstate)
            # Initialize state clean to be augmented next step.
            if len(nextstate_copy.shape) == 1:
                nextstate_copy.unsqueeze_(0)
            state = nextstate_copy[not_done]
            if len(state.shape) == 1:
                state.unsqueeze_(0)
            print("Remaining = {}".format(np.round(state.shape[0]) / start_states.shape[0], 2))
            var += uncert ** 2
        print("Finished rollouts: all terminated after %s timesteps" % (t))
        print("Added {} transitions to agent replay pool".format(transition_count))
        print("Agent replay pool: {}/{}".format(len(self.agent.replay_pool), self.agent.replay_pool.capacity))

    def _train_agent(self,IRL=False):
        if self._augment_offline_data:
            print("Augmenting model data with RAD")
        if IRL:
            self.agent.optimize(n_updates=self._policy_update_steps, env_pool=self.model.model.memory,
                            env_ratio=self._real_sample_ratio, augment_data=self._augment_offline_data,reward_function=self.reward_function)
        else:
            self.agent.optimize(n_updates=self._policy_update_steps, env_pool=self.model.model.memory,
                            env_ratio=self._real_sample_ratio, augment_data=self._augment_offline_data)

            
    def _train_reward(self):
        
        sample_pool = ReplayPool(capacity=1e6)
        init_state,traj_state,traj_action=self.expert_pool.sample_traj(self._reward_step)
        #init_state=np.array(init_state)
        init_state=torch.FloatTensor(np.array(init_state)).to(self._device)
        
        self._rollout_model_expert(init_state,sample_pool)
        samples = sample_pool.sample_all()
        
        state_batch = np.array([i for arr in samples.state for i in arr])
        action_batch = np.array([i for arr in samples.action for i in arr])
        
        agent_state_batch = torch.FloatTensor(state_batch).to(self._device)
        agent_action_batch = torch.FloatTensor(action_batch).to(self._device)
        
        
        traj_state=traj_state.reshape(-1,traj_state.shape[2])
        traj_action=traj_action.reshape(-1,traj_action.shape[2])
        expert_state_batch = torch.FloatTensor(traj_state).to(self._device)
        expert_action_batch = torch.FloatTensor(traj_action).to(self._device)
        
        agent_r=self.reward_function(torch.cat((agent_state_batch,agent_action_batch),1))
        expert_r=self.reward_function(torch.cat((expert_state_batch,expert_action_batch),1))
        #print(expert_r.shape)
        #avg_reward_norm = torch.mean(torch.squeeze(agent_r ** 2))
        
        #regularizer = avg_reward_norm
        loss=agent_r.mean()-expert_r.mean()
        #loss=agent_r.sum()/2000-expert_r.sum()/2000
        self.reward_opti.zero_grad()
        loss.backward()
        self.reward_opti.step()
                           
    def _load_population_models(self, params):
        model_dirs = params['population_model_dirs']
        print("Loading population on {} models".format(len(model_dirs)))
        self.population_models = {}
        for i, m in enumerate(model_dirs):
            pop_env = EnsembleGymEnv(params, self.model.real_env, self.model.eval_env)
            pop_env.model.load_model(m)
            # takes up RAM, let's remove
            del pop_env.model.memory_val
            del pop_env.model.memory
            self.population_models['model_{}'.format(i)] = pop_env
            print("Loaded model {}".format(m.split('/')[-1]))
        print("Finished loading {} population models".format(len(model_dirs)))

    def train_epoch(self, init=False):
        timesteps = 0
        error = None
        env = self.model.real_env
        collect_steps = self._init_collect if init else self._max_steps
        while timesteps < collect_steps:
            done = False
            # check if we were actually mid-rollout at the end of the last epoch
            if self._done:
                state = env.reset()
                self._curr_rollout = []
                self._env_step = 0
            else:
                state = self._curr_rollout[-1].nextstate
            while (not done) and (timesteps < collect_steps):
                if init:
                    action = env.action_space.sample()
                else:
                    action = self.agent.get_action(state, oac=self._oac)
                nextstate, reward, done, _ = env.step(action)
                self._env_step += 1
                # Check if environment actually terminated or just ran out of time
                if done and self._env_step != env.spec.max_episode_steps:
                    real_done = True
                else:
                    real_done = False
                t = Transition(state, action, reward, nextstate, real_done)
                self._curr_rollout.append(t)
                timesteps += 1
                if (timesteps) % 100 == 0:
                    print("Collected Timesteps: %s" % (timesteps))
                if done:
                    self._push_trajectory()
                state = nextstate
                self._done = done
                if not init:
                    if timesteps % self._model_train_freq == 0:
                        self._train_model()
                    if timesteps % self._rollout_model_freq == 0:
                        self._rollout_model()
                    if timesteps % self._train_policy_every == 0:
                        self._train_agent()


        if init:
            self._train_model()
            self._rollout_model()
        else:
            self._n_epochs += 1

        errors = [self.model.model.models[i].get_acquisition(self._curr_rollout, self.model.model.state_filter,
                                                             self.model.model.action_filter)
                  for i in range(len(self.model.model.models))]
        error = np.sqrt(np.mean(np.array(errors) ** 2))
        print("\nMSE Loss on latest rollout: %s" % error)
        steps_k_used = self._cur_steps_k
        self._steps_k_update()
        return timesteps, error, steps_k_used

    def _push_trajectory(self):
        collect_steps = len(self._curr_rollout)
        # randomly allocate the data to train and validation
        train_val_ind = random.sample(range(collect_steps), collect_steps)
        num_valid = int(np.floor(self.model.model.train_val_ratio * collect_steps))
        train_ind = train_val_ind[num_valid:]
        for i, t in enumerate(self._curr_rollout):
            self.model.update_state_filter(t.state)
            self.model.update_action_filter(t.action)
            if i in train_ind:
                self.model.model.add_data(t)
            else:
                self.model.model.add_data_validation(t)
        print("\nAdded {} samples for train, {} for valid".format(str(len(train_ind)),
                                                                  str(len(train_val_ind) - len(train_ind))))

    def train_offline(self, num_epochs, save_model=False, save_policy=False, load_model_dir=None):
        timesteps = 0
        val_size = 0
        train_size = 0

        # d4rl stuff - load all the offline data and train
        env = self.model.real_env
        # dataset = d4rl.qlearning_dataset(env, limit=5000)
        if self._params['env_name'] != 'AntMOPOEnv':
            dataset = d4rl.qlearning_dataset(env)
        else:
            with open('/Meta-Offline-RL/ant_mopo_1m_dataset.pkl', 'rb') as f:
                dataset = pickle.load(f)

        N = dataset['rewards'].shape[0]
        rollout = []

        if load_model_dir or self._params['ensemble_replace_model_dirs']:
            # Load pretrained model, overrride this if population loading flag is on
            if not self._params['ensemble_replace_model_dirs']:
                errors = self.model.model.load_model(load_model_dir)
            else:
                print(self._params['ensemble_replace_model_dirs'])
                errors = self.model.model.load_model_from_population(self._params['ensemble_replace_model_dirs'])
        else:
            self.model.update_state_filter(dataset['observations'][0])

            for i in range(N):
                state = dataset['observations'][i]
                action = dataset['actions'][i]
                nextstate = dataset['next_observations'][i]
                reward = dataset['rewards'][i]
                done = bool(dataset['terminals'][i])

                t = Transition(state, action, reward, nextstate, done)
                rollout.append(t)

                self.model.update_state_filter(nextstate)
                self.model.update_action_filter(action)

                # Do this probabilistically to avoid maintaining a huge array of indices
                if random.uniform(0, 1) < self.model.model.train_val_ratio:
                    self.model.model.add_data_validation(t)
                    val_size += 1
                else:
                    self.model.model.add_data(t)
                    train_size += 1
                timesteps += 1

            self._done = True

            print("\nAdded {} samples for train, {} for valid".format(str(train_size), str(val_size)))

            if save_model:
                print('Saving model!')

            self._train_model(d4rl_init=True, save_model=save_model)

        if self._params['morel']:
            self._morel_threshold = self._get_morel_threshold()
            self.model.model.set_morel_hparams(self._morel_threshold, self._morel_halt_reward)
        else:
            self._morel_threshold = None

        rewards, rewards_m, k_used, mopo_lam, myopic_wm, myopic_pop, rewards_pop, myopic_pop_worst, myopic_wm_worst = [], [], [], [], [], [], [], [], []
        for i in range(num_epochs):
            self._rollout_model(Penalty_only=True)
            self._train_agent(IRL=True)
            if i % 20 == 0 and not self.transfer:
                self._train_reward()
                #name='/home/luoqijun/code/IRL_Code/MBIRL/rethinking-code-supp/reward_hopper/medexp_hopper_reward_'+str(i)+str(self._seed)+'_'+'.pt'
                #torch.save(self.reward_function.state_dict(), name)
            
            reward_model = self.test_agent(use_model=True, n_evals=10)
            reward_actual_stats = self.test_agent(use_model=False)
            print("------------------------")
            stats_fmt = "{:<20}{:>30}"
            stats_str = ["Epoch",
                         "WM Reward Mean",
                         "WM Reward Max",
                         "WM Reward Min",
                         "WM Reward StdDev",
                         "True Reward Mean",
                         "True Reward Max",
                         "True Reward Min",
                         "True Reward StdDev"]
            stats_num = [i,
                         reward_model.mean().round(2),
                         reward_model.max().round(2),
                         reward_model.min().round(2),
                         reward_model.std().round(2),
                         reward_actual_stats.mean().round(2),
                         reward_actual_stats.max().round(2),
                         reward_actual_stats.min().round(2),
                         reward_actual_stats.std().round(2)]
            if hasattr(self, "population_models"):
                reward_pop = self.test_agent_population(full_trajectories=True, n_evals=10)
                reward_model_myopic, reward_pop_myopic = self.test_agent_population(full_trajectories=False)
                reward_model_myopic_worst, reward_pop_myopic_worst = self.test_agent_population(full_trajectories=False,
                                                                                                bad_states=True)
                pop_str = ["WM Myopic Mean", "WM Myopic Mean Worst"]
                pop_num = [reward_model_myopic.mean().round(2), reward_model_myopic_worst.mean().round(2)]
                for j, (stat, stat_myopic, stat_myopic_worst) in enumerate(
                        zip(reward_pop, reward_pop_myopic, reward_pop_myopic_worst)):
                    pop_str += ["Pop WM {} Mean".format(j),
                                "Pop WM {} Max".format(j),
                                "Pop WM {} Min".format(j),
                                "Pop WM {} StdDev".format(j),
                                "Pop WM {} Myopic Mean".format(j),
                                "Pop WM {} Myopic Mean Worst".format(j)]
                    pop_num += [stat.mean().round(2),
                                stat.max().round(2),
                                stat.min().round(2),
                                stat.std().round(2),
                                stat_myopic.mean().round(2),
                                stat_myopic_worst.mean().round(2)]
                stats_str.extend(pop_str)
                stats_num.extend(pop_num)
                myopic_wm.append(reward_model_myopic.mean())
                myopic_pop.append([s.mean() for s in reward_pop_myopic])
                rewards_pop.append([s.mean() for s in reward_pop])
                myopic_wm_worst.append([s.mean() for s in reward_model_myopic_worst])
                myopic_pop_worst.append([s.mean() for s in reward_pop_myopic_worst])
            for s, n in zip(stats_str, stats_num):
                print(stats_fmt.format(s, n))
            print("------------------------")
            # Log to csv (offline)
            rewards.append(reward_actual_stats.mean())
            rewards_m.append(reward_model.mean())
            k_used.append(self._cur_steps_k)
            if self._params['tune_mopo_lam']:
                ml = self.model.model.log_mopo_lam.exp().item()
            else:
                ml = self.model.model.mopo_lam
            mopo_lam.append(ml)
            save_stats = {'Reward': rewards, 'Reward_WM': rewards_m, 'k_used': k_used, 'mopo_lam': mopo_lam}
            if hasattr(self, "population_models"):
                save_stats['Myopic WM'] = myopic_wm
                save_stats['Myopic Population'] = myopic_pop
                save_stats['Rewards Population'] = rewards_pop
                save_stats['Myopic WM Worst'] = myopic_wm_worst
                save_stats['Myopic Population Worst'] = myopic_pop_worst
            df = pd.DataFrame(save_stats)
            lam = ['Adaptive' if self._params['adapt'] == 1 else 'fixed{}'.format(str(self._params['lam']))][0]
            save_name = "{}_{}_resid{}_{}_{}_offline".format(self._params['env_name'], lam, str(self._params['pca']),
                                                             self._params['filename'],
                                                             str(self._params['seed']))
            if self._params['comment']:
                save_name = save_name + '_' + self._params['comment']
            save_name += '.csv'
            df.to_csv(save_name)

            if save_policy and i % 20 == 0:
                save_path = './model_saved_weights_seed{}'.format(self._params['seed'])
                check_or_make_folder(save_path)
                print("Saving policy trained offline")
                self.agent.save_policy(
                    # "{}".format(self.model.model._model_id),
                    save_path,
                    num_epochs=i,
                    rew=int(reward_actual_stats.mean())
                )

        if not load_model_dir:
            errors = [self.model.model.models[i].get_acquisition(rollout[:1000], self.model.model.state_filter,
                                                                 self.model.model.action_filter)
                      for i in range(len(self.model.model.models))]
        error = np.sqrt(np.mean(np.array(errors) ** 2))
        print("\nMSE Loss on offline rollouts: %s" % error)
        steps_k_used = self._cur_steps_k
        self._steps_k_update()

        return timesteps, error, steps_k_used, rewards

    def _steps_k_update(self):
        if isinstance(self._steps_k, int):
            return
        else:
            steps_min, steps_max, start_epoch, end_epoch = self._steps_k
            m = (steps_max - steps_min) / (end_epoch - start_epoch)
            c = steps_min - m * start_epoch
        new_steps_k = m * self._n_epochs + c
        new_steps_k = int(min(steps_max, max(new_steps_k, steps_min)))
        if new_steps_k == self._cur_steps_k:
            return
        else:
            print("\nChanging model step size, going from %s to %s" % (self._cur_steps_k, new_steps_k))
            self._cur_steps_k = new_steps_k
            new_pool_size = int(
                self._cur_steps_k * self._num_rollouts * (
                        self._max_steps / self._model_train_freq) * self._model_retain_epochs)
            print("\nReallocating agent pool, going from %s to %s" % (self.agent.replay_pool.capacity, new_pool_size))
            self.agent.reallocate_replay_pool(new_pool_size)

    def get_pessimistic_states(self):
        all_states = self.model.model.memory.sample_all()
        all_states, _, all_rewards, _, _ = all_states
        all_states, all_rewards = np.array(all_states), np.array(all_rewards)
        worst_states_idx = all_rewards.argsort()[:5000]
        worst_states = all_states[worst_states_idx]
        return torch.FloatTensor(worst_states).to(device)

    def test_agent_population(self, full_trajectories=True, n_evals=5, bad_states=False):
        if not hasattr(self, "pessimistic_states"):
            self.pessimistic_states = self.get_pessimistic_states()
        if full_trajectories:
            return [self.test_agent(use_model=True, n_evals=n_evals, population_idx=model) for model in
                    self.population_models]
        else:
            # Do random sampling of 5000 states and rolling out 25 steps
            if not bad_states:
                start_states = torch.FloatTensor(
                    np.array(self.model.model.memory.sample(5000, unique=False)[0])).to(self._device)
            else:
                start_states = self.pessimistic_states
            # Need to test on self as we don't know how well we perform here under normal conditions
            own_WM_rewards = self.test_agent_myopic(start_states)
            # Now test on population
            pop_WM_rewards = [self.test_agent_myopic(start_states, population_idx=m) for m in self.population_models]
            return own_WM_rewards, pop_WM_rewards

    def test_agent_myopic(self, start_states, num_steps=100, population_idx=None):
        if population_idx:
            print("Getting myopic returns on populations models")
            test_env = self.population_models[population_idx]
        else:
            print("Getting myopic returns on World Model we trained in")
            test_env = self.model
        state = start_states
        sum_rewards = np.zeros(start_states.shape[0])
        done_false = [False for _ in range(start_states.shape[0])]
        # needed to subset the rewards properly
        idxs_remaining = np.arange(start_states.shape[0])
        t = 0
        test_env.convert_filter_to_torch()
        while t < num_steps:
            t += 1
            with torch.no_grad():
                # Get deterministic action
                _, _, action_det = self.agent.policy(state)
                nextstate, reward = test_env.model.random_env_step(state,
                                                                   action_det,
                                                                   get_var=self._keep_logvar,
                                                                   deterministic=self._deterministic,
                                                                   disable_mopo=True
                                                                   )
                if self._keep_logvar:
                    nextstate, nextstate_var, reward, reward_var = nextstate[0], nextstate[1], reward[0], reward[1]
                if self._is_done_func:
                    done = self._is_done_func(nextstate).cpu().numpy().tolist()
                else:
                    done = done_false[:nextstate.shape[0]]
                not_done = ~np.array(done)
                if self._reward_head:
                    reward = reward.cpu().detach().numpy()
                else:
                    reward = torch_reward(self.model.name, nextstate, action_det, done)
                sum_rewards[idxs_remaining] += reward
                idxs_remaining = idxs_remaining[not_done]
                if len(nextstate.shape) == 1:
                    nextstate.unsqueeze_(0)
                state = nextstate[not_done]
                if len(state.shape) == 1:
                    state.unsqueeze_(0)
                if not_done.sum() == 0:
                    break
        return sum_rewards

    def test_agent(self, use_model=False, n_evals=None, population_idx=None):
        if not use_model:
            assert population_idx is None, "You are evaluating performance on a real environment, why are you specifying population index?"
        rollout_rewards = []
        n_evals = n_evals if n_evals else self._n_eval_rollouts
        if use_model:
            if population_idx:
                test_env = self.population_models[population_idx]
            else:
                test_env = self.model
        else:
            test_env = self.model.eval_env
        for _ in range(n_evals):
            total_reward = 0
            time_step = 0
            done = False
            state = test_env.reset()
            while not done:
                time_step += 1
                if self.agent.augment_sac and self.agent.context_type == 'rad_augmentation':
                    state = np.concatenate((state, np.ones_like(state)))
                elif self.agent.augment_sac and self.agent.context_type == 'rad_magnitude':
                    if len(state) == 1:
                        state = state[0]
                    state = np.concatenate((state, np.ones((1,))))
                action = self.agent.get_action(state, deterministic=True)
                state, reward, done, info = test_env.step(action)
                if (self._params['env_name'] == 'AntMOPOEnv') and (not use_model):
                    reward = info['reward_angle'] + info['reward_ctrl'] + 1
                reward = 0 if reward is None else reward
                total_reward += reward
            rollout_rewards.append(total_reward)
        rollout_rewards = np.array(rollout_rewards)
        return rollout_rewards

    def modify_online_training_params(self, online_params):
        """
        Method to reassign the important training hyperparams to online, as offline hyperparams are different
        """
        self._num_rollouts = online_params['num_rollouts_per_step'] * online_params['model_train_freq']
        self._model_retain_epochs = online_params['model_retain_epochs']
        self._steps_k = online_params['steps_k']
        self._policy_update_steps = online_params['policy_update_steps']
        self._model_train_freq = online_params['model_train_freq']
        self._rollout_model_freq = online_params['rollout_model_freq']
        self._train_policy_every = online_params['train_policy_every']

    def _load_model_buffer_into_policy(self, new_buffer_size=None):
        """
        Method to load model replay buffer into the policy (seed for model-free training)
        """
        memory, memory_val = self.model.model.memory, self.model.model.memory_val
        if not new_buffer_size:
            new_buffer_size = int(len(memory) + len(memory_val))
        new_pool = ReplayPool(capacity=new_buffer_size)
        train_transitions = memory.get_all()
        val_transitions = memory_val.get_all()
        all_transitions = train_transitions + val_transitions
        for t in all_transitions:
            new_pool.push(t)
        print("Reallocating policy replay buffer as world model memory")
        self.agent.replay_pool = new_pool

    def train_policy_model_free(self, n_random_actions=0, update_timestep=1, n_collect_steps=0, log_interval=1000,
                                use_model_buffer=True, total_steps=3e6, policy_buffer_size=None, clear_buffer=False,
                                use_modified_env=False, horizon=None):
        """
        Method to train the internal policy in a model-free setting
        """

        if not use_modified_env:
            env = self.model.eval_env
        else:
            # Using modified environment! TODO: make this more general, this is just half-cheetah right now.
            from modified_envs import HalfCheetahEnv
            print('Using modified environments!')
            env = HalfCheetahEnv()
            horizon = 1000
        agent = self.agent

        if use_model_buffer and not clear_buffer:
            self._load_model_buffer_into_policy(new_buffer_size=policy_buffer_size)

        if clear_buffer:
            self.agent.replay_pool = ReplayPool(capacity=1e6)
            # Recollect 5000 new transitions.
            n_collect_steps = 5000

        avg_length = 0
        time_step = 0
        cumulative_timestep = 0
        cumulative_log_timestep = 0
        n_updates = 0
        i_episode = 0
        log_episode = 0
        samples_number = 0
        episode_rewards = []
        episode_steps = []
        all_rewards = []
        all_timesteps = []
        all_lengths = []

        random.seed(self._seed)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        env.seed(self._seed)
        env.action_space.np_random.seed(self._seed)

        max_steps = horizon if horizon is not None else env.spec.max_episode_steps

        while samples_number < total_steps:
            time_step = 0
            episode_reward = 0
            i_episode += 1
            log_episode += 1
            state = env.reset()
            done = False
            while (not done):
                cumulative_log_timestep += 1
                cumulative_timestep += 1
                time_step += 1
                samples_number += 1
                if samples_number < n_random_actions:
                    action = env.action_space.sample()
                else:
                    action = agent.get_action(state)
                nextstate, reward, done, _ = env.step(action)
                # Terminate if over horizon
                if horizon is not None and time_step == horizon:
                    done = True
                # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
                real_done = False if time_step == max_steps else done
                agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
                state = nextstate
                episode_reward += reward
                # update if it's time
                if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                    q1_loss, q2_loss, pi_loss, a_loss = agent.optimize(update_timestep,
                                                                       augment_data=self._augment_offline_data)
                    n_updates += 1
                # logging
                if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
                    avg_length = np.mean(episode_steps)
                    running_reward = np.mean(episode_rewards)
                    eval_reward = self.test_agent(n_evals=1).mean()
                    print(
                        'Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(
                            i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
                    episode_steps = []
                    episode_rewards = []
                    all_timesteps.append(cumulative_timestep)
                    all_rewards.append(eval_reward)
                    all_lengths.append(avg_length)
                    df = pd.DataFrame(
                        {'Timesteps': all_timesteps, 'Reward': all_rewards, 'Average_Length': all_lengths})
                    save_name = "model_free_{}_seed{}".format(self._params['env_name'], str(self._params['seed']))
                    save_name += '.csv'
                    df.to_csv(save_name)

            episode_steps.append(time_step)
            episode_rewards.append(episode_reward)

    def _get_morel_threshold(self):
        """ Uses a UCB heuristic similar to author's paper to calculate the threshold except with robust statistics """
        # Interpolates between the median and the 99th percentile penalty values of the offline data
        preds = self.model.model.get_replay_buffer_predictions(only_validation=True, return_sample=True)
        sample, mu, logvar = preds.chunk(3, dim=2)

        sample_nextstates, sample_rewards, states_mu, rewards_mu, logvar_states, logvar_rewards = sample[:,:,:-1], sample[:,:,-1], mu[:,:,:-1], mu[:,:,-1], logvar[:,:,:-1], logvar[:,:,-1]

        allocation = torch.randint(0, self.model.model.num_elites, (preds.shape[1],), device=device)
        allocation = torch.tensor([self.model.model._elites_idx[idx] for idx in allocation]).to(device)
        allocation_states = allocation.repeat(sample_nextstates.shape[2], 1).T.view(1, -1, sample_nextstates.shape[2])
        allocation_rewards = allocation.view(1, -1, 1)

        nextstates = sample_nextstates.gather(0, allocation_states).squeeze()
        rewards = sample_rewards.unsqueeze(2).gather(0, allocation_rewards).squeeze()

        penalties = self.model.model.get_penalty(self._params['mopo_penalty_type'], nextstates, sample_nextstates, logvar_states, rewards, sample_rewards, logvar_rewards, states_mu, rewards_mu, allocation)

        penalties_median, penalties_p99 = penalties.median(), penalties.quantile(.99)

        penalties_mad = (penalties - penalties_median).abs().median()

        beta_max = (penalties_p99 - penalties_median) / penalties_mad

        beta = beta_max * self._params['morel_thresh']

        return (penalties_median + beta * penalties_mad).item()
