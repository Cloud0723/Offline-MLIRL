import argparse
import os
import random
import time
import pdb

import gym
from gym.wrappers import TimeLimit
import numpy as np
import pandas as pd
import yaml
import torch
import d4rl

from model import EnsembleGymEnv
from done_funcs import hopper_is_done_func, walker2d_is_done_func, ant_is_done_func
from sac import SAC_Agent
from train_funcs import (collect_data, test_agent, train_agent,
                         train_agent_model_free, train_agent_model_free_debug)
from utils import MeanStdevFilter, reward_func
from trainer import Trainer

import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def between_0_1(x):
    x = float(x)
    if x > 1 or x < 0:
        raise argparse.ArgumentTypeError("This should be between 0 and 1 (inclusive)")
    return x


def train_agent_new(params, online_yaml_config=None):
    params['zeros'] = False

    if params['reward_head']:
        env = gym.make(params['env_name'])
        eval_env = gym.make(params['env_name'])

    else:
        raise Exception('Environment Not Supported!')

    env_name_lower = params['env_name'].lower()

    if 'hopper' in env_name_lower:
        params['is_done_func'] = hopper_is_done_func
    elif 'walker' in env_name_lower:
        params['is_done_func'] = walker2d_is_done_func
    elif 'ant' in env_name_lower:
        params['is_done_func'] = ant_is_done_func
    else:
        params['is_done_func'] = None

    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]

    env=EnsembleGymEnv(params, env, eval_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    env.real_env.seed(params['seed'])
    env.eval_env.seed(params['seed'] + 1)
    env.real_env.action_space.seed(params['seed'])
    env.eval_env.action_space.seed(params['seed'] + 1)
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    if isinstance(params['steps_k'], list):
        init_steps_k = params['steps_k'][0]
    else:
        init_steps_k = params['steps_k']

    steps_per_epoch = params['epoch_steps'] if params['epoch_steps'] else env.real_env.env.spec.max_episode_steps

    if params['d4rl']:
        # Trying this option out right now.
        init_buffer_size = init_steps_k * params['num_rollouts_per_step'] * steps_per_epoch * params[
            'model_retain_epochs']
        print('Initial Buffer Size: {} using model_retain_epochs={}'.format(init_buffer_size,
                                                                            params['model_retain_epochs']))
    else:
        init_buffer_size = init_steps_k * params['num_rollouts_per_step'] * steps_per_epoch
        print('Initial Buffer Size: {}'.format(init_buffer_size))

    agent = SAC_Agent(params['seed'], state_dim, action_dim, gamma=params['gamma'], buffer_size=init_buffer_size,
                      target_entropy=params['target_entropy'], augment_sac=params['augment_sac'],
                      rad_rollout=params['rad_rollout'], context_type=params['context_type'])



    trainer = Trainer(params, env, agent, device=device)

    total_timesteps = 0
    rewards, rewards_m, lambdas, steps_used, k_used, errors, varmean, samples = [], [], [], [], [], [], [], []
    if params['d4rl']:
        print("\nRunning initial training with offline data...")
        timesteps, error, model_steps, rewards = trainer.train_offline(params['offline_epochs'],
                                                                       save_model=params['save_model'],
                                                                       save_policy=params['save_policy'],
                                                                       load_model_dir=params['load_model_dir'],
                                                                       )
        total_timesteps += timesteps
        varmean.append(trainer.var_mean)
    else:
        print("\nCollecting random rollouts...")
        timesteps, error, model_steps = trainer.train_epoch(init=True)
        total_timesteps += timesteps
        varmean.append(trainer.var_mean)

    return rewards[-10:]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default='HalfCheetah-v2')  ## only works properly for HalfCheetah and Ant
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--num_models', '-nm', type=int, default=7)
    parser.add_argument('--adapt', '-ad', type=int, default=0)  ## set to 1 for adaptive
    # parser.add_argument('--steps', '-s', type=int, default=100)  ## maximum time we step through an env per episode
    parser.add_argument('--steps_k', '-sk', type=int,  # nargs='+',
                        default=1)  ## maximum time we step through an env to make artificial rollouts
    parser.add_argument('--reward_steps', '-rs', type=int,  # nargs='+',
                        default=10)  ## maximum time we step through an env to make artificial rollouts to update reward estimator
    parser.add_argument('--outer_steps', '-in', type=int,
                        default=3000)  ## how many time steps/samples we collect each outer loop (including initially)
    parser.add_argument('--max_timesteps', '-maxt', type=int, default=6000)  ## total number of timesteps
    parser.add_argument('--model_epochs', '-me', type=int, default=2000)  ## max number of times we improve model
    parser.add_argument('--update_timestep', '-ut', type=int,
                        default=50000)  ## for PPO only; how many steps to accumulate before training on them
    parser.add_argument('--policy_iters', '-it', type=int, default=1000)  ## max number of times we improve policy
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    parser.add_argument('--gamma', '-gm', type=float, default=0.99)
    parser.add_argument('--lam', '-la', type=float, default=0)
    parser.add_argument('--pca', '-pc', type=float, default=0)  ## threshold for residual to stop, try [1e-4,2-e4]
    parser.add_argument('--sigma', '-si', type=float, default=0.01)
    parser.add_argument('--filename', '-f', type=str, default='ModelBased')
    parser.add_argument('--dir', '-d', type=str, default='data')
    parser.add_argument('--yaml_file', '-yml', type=str, default=None)
    parser.add_argument('--uuid', '-id', type=str, default=None)
    parser.add_argument('--fix_std', dest='fix_std', action='store_true')
    parser.add_argument('--var_type', type=str, default='reward', choices=('reward', 'state'))
    parser.add_argument('--states', type=str, default='uniform', choices=('uniform', 'start', 'entropy'))
    parser.add_argument('--reward_head', '-rh', type=int, default=1)  # 1 or 0
    parser.add_argument('--model_free', dest='model_free', action='store_true')
    parser.add_argument('--var_max', dest='var_max', action='store_true')
    parser.add_argument('--no_logvar_head', dest='logvar_head', action='store_false')
    parser.add_argument('--comment', '-c', type=str, default=None)
    parser.add_argument('--policy_update_steps', type=int, default=40)
    parser.add_argument('--init_collect', type=int, default=5000)
    parser.add_argument('--train_policy_every', type=int, default=1)
    parser.add_argument('--num_rollouts_per_step', type=int, default=400)
    parser.add_argument('--n_eval_rollouts', type=int, default=10)
    parser.add_argument('--train_val_ratio', type=float, default=0.2)
    parser.add_argument('--real_sample_ratio', type=float, default=0.05)
    parser.add_argument('--model_train_freq', type=int, default=250)
    parser.add_argument('--rollout_model_freq', type=int, default=250)
    parser.add_argument('--oac', type=bool, default=False)
    parser.add_argument('--espi', type=bool, default=False)
    parser.add_argument('--num_elites', type=int, default=5)
    parser.add_argument('--var_thresh', type=float, default=100)
    parser.add_argument('--epoch_steps', type=int, default=None)
    parser.add_argument('--target_entropy', type=float, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--d4rl', dest='d4rl', action='store_true')
    parser.add_argument('--train_memory', type=int, default=800000)
    parser.add_argument('--val_memory', type=int, default=200000)
    parser.add_argument('--mopo', dest='mopo', action='store_true')
    parser.add_argument('--morel', dest='morel', action='store_true')
    # MOPO/MOReL tuning parameters
    parser.add_argument('--mopo_lam', type=float, default=1)
    parser.add_argument('--morel_thresh', type=between_0_1, default=0.3)
    parser.add_argument('--morel_halt_reward', type=float, default=-10)
    # This basically says to not truncate rollouts, but to keep going (like M2AC Non-Stop mode)
    parser.add_argument('--morel_non_stop', type=bool, default=False)
    parser.add_argument('--tune_mopo_lam', dest='tune_mopo_lam', action='store_true')
    parser.add_argument('--mopo_penalty_type', type=str, default='mopo_default', choices=(
    'mopo_default', 'ensemble_var', 'ensemble_std', 'ensemble_var_rew', 'ensemble_var_comb', 'mopo_paper', 'lompo', 'm2ac', 'morel'))
    parser.add_argument('--mopo_uncertainty_target', type=float, default=1.5)
    
    parser.add_argument('--offline_epochs', type=int, default=1000)
    parser.add_argument('--model_retain_epochs', type=int, default=100)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--transfer', type=bool, default=False)
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--deterministic_rollouts', type=bool, default=False)
    # Needed as some models seem to early terminate (this happens in author's code too, so not a PyTorch issue)
    parser.add_argument('--min_model_epochs', type=int, default=None)
    parser.add_argument('--augment_offline_data', type=bool, default=False)
    parser.add_argument('--augment_sac', type=bool, default=False)
    parser.add_argument('--context_type', type=str, default='rad_augmentation')
    parser.add_argument('--rad_rollout', type=bool, default=False)
    parser.add_argument('--save_policy', type=bool, default=False)
    parser.add_argument('--population_model_dirs', type=str, default=[], nargs="*")
    parser.add_argument('--ensemble_replace_model_dirs', type=str, default=[], nargs="*")
    parser.add_argument('--l2_reg_multiplier', type=float, default=1.)
    parser.add_argument('--model_lr', type=float, default=0.001)
    parser.set_defaults(fix_std=False)
    parser.set_defaults(model_free=False)
    parser.set_defaults(var_max=False)
    parser.set_defaults(logvar_head=True)

    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]

    online_yaml_config = None

    assert isinstance(params['steps_k'], (int, list)), "must be either a single input or a collection"

    if isinstance(params['steps_k'], list):
        assert len(params[
                       'steps_k']) == 4, "if a list of inputs, must have 4 inputs (start steps, end steps, start epoch, end epoch)"

    time.sleep(random.random())
    if not (os.path.exists(params['dir'])):
        os.makedirs(params['dir'])
    os.chdir(params['dir'])

    if params['uuid']:
        if not (os.path.exists(params['uuid'])):
            os.makedirs(params['uuid'])
        os.chdir(params['uuid'])

    rewards = train_agent_new(params, online_yaml_config)
    rewards = np.array(rewards)
    print(rewards)
    sys.stderr.write(str(np.mean(rewards)))

    return np.mean(rewards)


if __name__ == '__main__':
    main()
