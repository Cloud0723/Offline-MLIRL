from typing import List
import os
from random import shuffle
import glob

import numpy as np
import torch
import yaml
import gym
from gym.wrappers import TimeLimit
import d4rl
from numpy.random import default_rng

from modified_envs import AntMOPOEnv
from model import EnsembleGymEnv
from trainer import Trainer
from done_funcs import *
from sac import SAC_Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Assessor:
    """
    An assessor that will tell us how well a policy performs agains:
    a) each member of our world model collection
    b) our actual real world env
    """

    def __init__(self, trainer: Trainer, d4rl_env: str):
        """
        Constructor

        Args:
            trainer: A trainer instance that contains the WM used to train a policy, and "population" members
                    that we can assess against, as well as real envs
            d4rl_env: String of the d4rl offline env we've trained on so we can load start states
        """
        self._trainer = trainer
        env = gym.make(d4rl_env)
        self._dataset = d4rl.qlearning_dataset(env)
        self._rng = default_rng()

    def evaluate_policy(self, policy, wm_start_states, wm_init_states, real_evals=20, truncated_steps=100):
        """
        Method to evaluate a policy in our WMs, using both full traj and truncated/myopic

        Args:

            policy: The policy under evaluation

            wm_start_states: A NxM matrix of starting states for the WM 
                            (could be anywhere in the trajectory) where 
                            N is num states (i.e., evals), and M is state
                            dimension

            wm_init_states: A NxM matrix of initial states (s_0) for the
                            WM (sampled from initial state dist.) where N
                            is num states (i.e., evals), and M is state
                            dimension

            real_evals: How many returns to gather under true model (i.e., iterations)

            truncated_steps: How many steps to rollout for with the wm_start_states
        """

        stats = {
            "True Perf": None,
            "WM Perf": None,
            "WM Myopic Perf": None,
        }

        # Save old policy just in case...
        old_policy = self._trainer.agent

        # Load the new policy
        self._trainer.agent = policy

        # Step 1: Get actual performance on real env
        stats["True Perf"] = self._trainer.test_agent(use_model=False, n_evals=real_evals)

        # Step 2: Get full trajectory perf in the WMs assuming it's T=1000 (MuJoCo)
        stats["WM Perf"] = self._test_agent_wm(1000, wm_init_states)

        # Step 3: Get myopic perf in the WMs
        stats["WM Myopic Perf"] = self._test_agent_wm(truncated_steps, wm_start_states)

        # Let's reload that old policy
        self._trainer.agent = old_policy

        return stats

    def _test_agent_wm(self, num_steps, start_states):
        """
        Internal method to test our policy we loaded in the world models.

        Args:
            num_steps: Horizon to rollout for
            start_states: start states tensor
        """
        results = []

        for wm_idx in self._trainer.population_models:
            results.append(
                self._trainer.test_agent_myopic(start_states=start_states, num_steps=num_steps, population_idx=wm_idx))

        return np.array(results)

    def get_pool_start_states(self, n_samples: int, reverse_reward: bool = False, unique: bool = True):
        """
        Method to return some random states from the D4RL pool

        Args:
            n_samples: Int of number of samples
            reverse_reward: Whether subset reversed reward
            unique: Whether to have unique samples
        """
        if reverse_reward:
            assert unique is True, "Reversed reward is by definition unique"
            all_states, all_rewards = np.array(self._dataset['observations']), np.array(self._dataset['rewards'])
            worst_states_idx = all_rewards.argsort()[:n_samples]
            start_states = all_states[worst_states_idx]
        else:
            len_dataset = self._dataset['observations'].shape[0]
            idx = np.random.randint(0, len_dataset, n_samples) if not unique else self._rng.choice(len_dataset,
                                                                                                   size=n_samples,
                                                                                                   replace=False)
            start_states = self._dataset['observations'][idx]
        return torch.Tensor(start_states).to(device)


def get_assessor_from_yaml(yaml_file_path: str, wm_dirs: List[str], d4rl_env: str):
    with open(yaml_file_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['population_model_dirs'] = wm_dirs

    if params['env_name'] == 'AntMOPOEnv':
        env = TimeLimit(AntMOPOEnv(), 1000)
        eval_env = TimeLimit(AntMOPOEnv(), 1000)
    else:
        env = gym.make(params['env_name'])
        eval_env = gym.make(params['env_name'])

    env = EnsembleGymEnv(params, env, eval_env)

    env_name_lower = params['env_name'].lower()

    if isinstance(params['steps_k'], list):
        init_steps_k = params['steps_k'][0]
    else:
        init_steps_k = params['steps_k']

    steps_per_epoch = params['epoch_steps'] if params['epoch_steps'] else env.real_env.env.spec.max_episode_steps

    if 'hopper' in env_name_lower:
        params['is_done_func'] = hopper_is_done_func
    elif 'walker' in env_name_lower:
        params['is_done_func'] = walker2d_is_done_func
    elif 'ant' in env_name_lower:
        params['is_done_func'] = ant_is_done_func
    else:
        params['is_done_func'] = None

    init_buffer_size = init_steps_k * params['num_rollouts_per_step'] * steps_per_epoch * params[
        'model_retain_epochs']

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC_Agent(params['seed'], state_dim, action_dim, gamma=params['gamma'], buffer_size=init_buffer_size,
                      target_entropy=params['target_entropy'], augment_sac=params['augment_sac'],
                      rad_rollout=params['rad_rollout'], context_type=params['context_type'])

    trainer = Trainer(params, env, agent, device=device)

    return Assessor(trainer, d4rl_env)


def real_start_states(env, n_states, device='cuda'):
    """
    This just gets some real start states from the true gym environments
    """
    start_states = [env.reset() for _ in range(n_states)]
    return torch.Tensor(start_states).to(device)


def get_d4rl_start_states(d4rl_name: str, n_states: int, device='cuda'):
    """
    This is in fact broken, because D4RL doesn't save initial states in slot 0
    """
    d4rl_sequence = d4rl.sequence_dataset(gym.make(d4rl_name))
    starts = [t['observations'][0] for t in d4rl_sequence]
    shuffle(starts)
    return torch.Tensor(starts).to(device)[:n_states]


def get_policy_from_seed(seed: int, directory: str, env: gym.Env):
    """
    Returns a SAC policy from seed and policies directory
    """
    fname_match = directory + '/' + 'torch_policy_weights_{}*.pt'.format(seed)
    for f in glob.glob(fname_match):
        sac = SAC_Agent(1, env.observation_space.shape[0], env.action_space.shape[0])
        saved_policy = torch.load(f)
        sac.policy.load_state_dict(saved_policy['policy_state_dict'])
        return sac
    else:
        print("Policy not found!")
