import itertools
import math
import os
import random
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus
from torch.nn.init import _calculate_correct_fan

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TransitionContext = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'real_done', 'rad_context'))
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'real_done'))


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    p = p.cpu().numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x.to(device)


def truncated_normal(uniform):
    return parameterized_truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2)


def truncated_normal_replace(m, mode='fan_in'):
    fan = _calculate_correct_fan(m, mode)
    std = 1/(2*np.sqrt(fan))
    with torch.no_grad():
        weight = (truncated_normal(m.uniform_()))
        return weight * std


def truncated_normal_init(layer):
    if type(layer) in [nn.Linear]:
        layer.weight.data = truncated_normal_replace(layer.weight.data)


def reward_func(s1, s2, a, env_name, state_filter=None, is_done_func=None):
    if state_filter:
        s1_real = s1 * state_filter.stdev + state_filter.mean
        s2_real = s2 * state_filter.stdev + state_filter.mean
    else:
        s1_real = s1
        s2_real = s2
    if env_name == "HalfCheetah-v2":
        return np.squeeze(s2_real)[-1] - 0.1 * np.square(a).sum()
    if env_name == "Ant-v2":
        if is_done_func:
            if is_done_func(torch.Tensor(s2_real).reshape(1,-1)):
                return 0.0
        return np.squeeze(s2_real)[-1] - 0.5 * np.square(a).sum() + 1.0
    if env_name == "Swimmer-v2":
        return np.squeeze(s2_real)[-1] - 0.0001 * np.square(a).sum()
    if env_name == "Hopper-v2":
        if is_done_func:
            if is_done_func(torch.Tensor(s2_real).reshape(1,-1)):
                return 0.0
        return np.squeeze(s2_real)[-1] - 0.1 * np.square(a).sum() - 3.0 * np.square(s2_real[0] - 1.3) + 1.0


class MeanStdevFilter():
    def __init__(self, shape, clip=10.0):
        self.eps = 1e-12
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = 0
        self.stdev = 1

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
        self.stdev[self.stdev <= self.eps] = 1.0

    def reset(self):
        self.__init__(self.shape, self.clip)

    def update_torch(self):
        self.torch_mean = torch.FloatTensor(self.mean).to(device)
        self.torch_stdev = torch.FloatTensor(self.stdev).to(device)

    def filter(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def filter_torch(self, x: torch.Tensor):
        self.update_torch()
        return torch.clamp(((x - self.torch_mean) / self.torch_stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean

    def invert_torch(self, x: torch.Tensor):
        return (x * self.torch_stdev) + self.torch_mean


def tidy_up_weight_dir(guids=None):
    if guids == None:
        guids = []
    files = [i for i in os.listdir("./data/") if i.endswith("pth")]
    for weight_full in files:
        weight = weight_full.split('_')[1]
        if weight.split('.')[0] not in guids:
            os.remove("./data/" + weight_full)


def prepare_data(state, action, nextstate, state_filter, action_filter):
    state_filtered = state_filter.filter(state)
    action_filtered = action_filter.filter(action)
    state_action_filtered = np.concatenate((state_filtered, action_filtered), axis=1)
    delta = np.array(nextstate) - np.array(state)
    return state_action_filtered, delta


def get_residual(newdata, pca_data, pct=0.99):
    X_pca = np.array(pca_data)
    # standardize
    X_pca = (X_pca - np.mean(X_pca)) / (np.std(X_pca) + 1e-8)
    
    Q, Sigma, _ = np.linalg.svd(X_pca.T)
    # proportion
    weight = np.cumsum(Sigma / np.sum(Sigma))
    index = np.sum((weight > pct) == 0)
    train_resid = 1-weight[index]
    V = Q[:,:index+1]
    
    basis = V.dot(V.T)
    
    X = np.array(newdata)
    # standardize with respect to old data
    X = (X - np.mean(X_pca)) / (np.std(X_pca) + 1e-8)
    orig = X.T.dot(X)
    projected = np.matmul(np.matmul(basis, orig), basis)
    residual = (np.trace(orig) - np.trace(projected))/np.trace(orig)
    return(residual, train_resid)


def get_stats(env, state, action, state_filter, action_filter, done, dynamics=False, reward_head=0):
    with torch.no_grad():
        stats_mean = []
        stats_var = []
        for model in env.model.models.values():
            if model.model.is_probabilistic:
                nextstate, reward = model.get_next_state_reward(state, action, state_filter, action_filter, True)
                if dynamics:
                        raise Exception('Not Implemented')
                if reward_head:
                    stats_mean.append(reward[0])
                    stats_var.append(reward[1].exp())
                else:
                    # TODO: make this more efficient
                    reward = torch.tensor(torch_reward(env.name, nextstate[0], action, done), device=device)
                    stats_mean.append(reward)
                    stats_var.append(nextstate[1][:,-1].exp())
            else:
                nextstate, reward = model.get_next_state_reward(state, action, state_filter, action_filter, False)
                if dynamics:
                    stats_mean.append(nextstate)
                    stats_var.append(torch.zeros(nextstate.shape, device=device))
                if reward_head:
                    stats_mean.append(reward)
                    stats_var.append(torch.zeros(reward.shape, device=device))
                else:
                    # TODO: make this more efficient
                    reward = torch.tensor(torch_reward(env.name, nextstate, action, done), device=device)
                    stats_mean.append(reward)
                    stats_var.append(torch.zeros(reward.shape, device=device))
        if dynamics:
            return (torch.stack(stats_mean) - torch.stack(stats_mean).mean((0))).pow(2).sum(2).mean(0).detach().cpu().numpy()
        else:
            # equivalent to the Lakshminarayanan paper
            return torch.sqrt(torch.var(torch.stack(stats_mean), axis=0) + torch.mean(torch.stack(stats_var), axis=0)).detach().cpu().numpy()


def random_env_forward(data, env, reward_head):
    """Randomly allocate the data through the different dynamics models"""
    y = torch.zeros((data.shape[0], env.observation_space.shape[0]+reward_head), device=device)
    allocation = torch.randint(0, len(env.model.models), (data.shape[0],))
    for i in env.model.models:
        data_i = data[allocation == i]
        y_i, _ = env.model.models[i].forward(data_i)
        y[allocation == i] = y_i
    return y


def filter_torch(x, mean, stddev):
    x_f = (x - mean) / stddev
    return torch.clamp(x_f, -3, 3)


def filter_torch_invert(x_f, mean, stddev):
    x = (x_f * stddev) + mean
    return x


def halfcheetah_reward(nextstate, action):
    return (nextstate[:,-1] - 0.1 * torch.sum(torch.pow(action, 2), 1)).detach().cpu().numpy()


def ant_reward(nextstate, action, dones):
    reward = (nextstate[:,-1] - 0.5 * torch.sum(torch.pow(action, 2), 1) + 1.0).detach().cpu().numpy()
    reward[dones] = 0.0
    return reward


def swimmer_reward(nextstate, action):
    reward = (nextstate[:,-1] - 0.0001 * torch.sum(torch.pow(action, 2), 1)).detach().cpu().numpy()
    return reward


def hopper_reward(nextstate, action, dones):
    reward = (nextstate[:,-1] - 0.1 * torch.sum(torch.pow(action, 2), 1) - 3.0 * (nextstate[:,0] - 1.3).pow(2) + 1.0).detach().cpu().numpy()
    reward[dones] = 0.0
    return reward


def torch_reward(env_name, nextstate, action, dones=None):
    if env_name == "HalfCheetah-v2":
        return halfcheetah_reward(nextstate, action)
    elif env_name == "Ant-v2":
        return ant_reward(nextstate, action, dones)
    elif env_name == "Hopper-v2":
        return hopper_reward(nextstate, action, dones)
    elif env_name == "Swimmer-v2":
        return swimmer_reward(nextstate, action)
    else:
        raise Exception('Environment not supported')


class GaussianMSELoss(nn.Module):

    def __init__(self):
        super(GaussianMSELoss, self).__init__()

    def forward(self, mu_logvar, target, logvar_loss = True):
        mu, logvar = mu_logvar.chunk(2, dim=1)
        inv_var = (-logvar).exp()
        if logvar_loss:
            return (logvar + (target - mu)**2 * inv_var).mean()
        else:
            return ((target - mu)**2).mean()


class FasterReplayPool:

    def __init__(self, action_dim, state_dim, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()

    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action': np.zeros((self.capacity, self._action_dim), dtype='float32'),
            'reward': np.zeros((self.capacity), dtype='float32'),
            'nextstate': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'real_done': np.zeros((self.capacity), dtype='bool')
        }

    def push(self, transition: Transition):

        # Handle 1-D Data
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity

        for key, value in transition._asdict().items():
            self._memory[key][idx] = value

        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k,v in self._memory.items()}
        return Transition(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(self._size, size=batch_size, replace=False)
        return self._return_from_idx(idx)

    def sample_all(self):
        return self._return_from_idx(np.arange(0, self._size))

    def get(self, start_idx, end_idx):
        raise NotImplementedError

    def get_all(self):
        raise NotImplementedError

    def _get_from_idx(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def initialise(self, old_pool):
        # Not Tested
        old_memory = old_pool.sample_all()
        for key in self._memory:
            self._memory[key] = np.append(self._memory[key], old_memory[key], 0)

class FasterReplayPoolCtxt:

    def __init__(self, action_dim, state_dim, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()

    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action': np.zeros((self.capacity, self._action_dim), dtype='float32'),
            'reward': np.zeros((self.capacity), dtype='float32'),
            'nextstate': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'real_done': np.zeros((self.capacity), dtype='bool'),
            'rad_context': np.zeros((self.capacity, self._state_dim), dtype='float32')
        }

    def push(self, transition: Transition):

        # Handle 1-D Data
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity

        for key, value in transition._asdict().items():
            self._memory[key][idx] = value

        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k,v in self._memory.items()}
        return TransitionContext(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(self._size, size=batch_size, replace=False)
        return self._return_from_idx(idx)

    def sample_all(self):
        return self._return_from_idx(np.arange(0, self._size))

    def get(self, start_idx, end_idx):
        raise NotImplementedError

    def get_all(self):
        raise NotImplementedError

    def _get_from_idx(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def initialise(self, old_pool):
        # Not Tested
        old_memory = old_pool.sample_all()
        for key in self._memory:
            self._memory[key] = np.append(self._memory[key], old_memory[key], 0)

class ReplayPool:

    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))
        
    def push(self, transition: Transition):
        """ Saves a transition """
        self._memory.append(transition)
        
    def sample(self, batch_size: int, unique: bool = True, dist=None) -> Transition:
        transitions = random.sample(self._memory, batch_size) if unique else random.choices(self._memory, k=batch_size)
        return Transition(*zip(*transitions))
    
    def sample_traj(self, truncate_length = 300):
        traj_num = len(self._memory)//1000 # number of trajectories
        init_state=[self._memory[i * 1000].state for i in range(traj_num)]

        traj_s,traj_a = [], []
        for i in range(traj_num):
            s,a=self.get2(i*1000, i*1000+truncate_length)
            traj_s.append(s)
            traj_a.append(a)
        traj_s=np.array(traj_s)
        traj_a=np.array(traj_a)
        return init_state,traj_s,traj_a 
    
    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return transitions
    
    def get2(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        states=np.array([i.state for i in transitions])
        actions=np.array([i.action for i in transitions])
        return states,actions

    def get_all(self) -> Transition:
        return self.get(0, len(self._memory))

    def sample_all(self) -> Transition:
        return Transition(*zip(*(self.get_all())))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

    def initialise(self, old_pool: 'ReplayPool'):
        old_memory = old_pool.get_all()
        self._memory.extend(old_memory)


class ReplayPoolCtxt:

    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))

    def push(self, transition: TransitionContext):
        """ Saves a transition """
        self._memory.append(transition)

    def sample(self, batch_size: int, unique: bool = True, dist=None) -> TransitionContext:
        transitions = random.sample(self._memory, batch_size) if unique else random.choices(self._memory, k=batch_size)
        return TransitionContext(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int) -> TransitionContext:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return transitions

    def get_all(self) -> TransitionContext:
        return self.get(0, len(self._memory))

    def sample_all(self) -> TransitionContext:
        return TransitionContext(*zip(*(self.get_all())))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

    def initialise(self, old_pool: 'ReplayPoolCtxt'):
        old_memory = old_pool.get_all()
        self._memory.extend(old_memory)


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))


def check_or_make_folder(folder_path):
    """
    Helper function that (safely) checks if a dir exists; if not, it creates it
    """
    
    folder_path = Path(folder_path)

    try:
        folder_path.resolve(strict=True)
    except FileNotFoundError:
        print("{} dir not found, creating it".format(folder_path))
        os.mkdir(folder_path)