# Understanding Expertise through Demonstrations:A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning

## Figures

## Installation
- PyTorch 1.13.1
- OpenAI Gym
- [MuJoCo210](https://www.roboti.us/license.html)
- DUE
- see more in requirements.txt


## File Structure
- Experiment data ：`data/`
- Configurations: `args_yml/`
- Expert Demonstrations: `expert_data/`

## Instructions
- All the experiments are to be run under the root folder. 
- After running, you will see the training logs in `data/` folder.

## Experiments
All the commands below are also provided in `run.sh`.

### Sec 1 Offline-IRL benchmark (MuJoCo)
Before experiment, you can download our expert demonstrations here and download our trained world model(otherwise it will be retrained by default).

```bash
python train.py --yaml_file args_yml/model_base_IRL/{env}.yml --seed 0 --uuid test # env is in {hopper, walker2d, halfcheetah}
```

### Sec 2 Transfer task
First, you can generate expert data by training expert policy:
Make sure that the `env_name` parameter in `configs/samples/experts/ant_transfer.yml` is set to `CustomAnt-v0`
```bash
python common/train_gd.py configs/samples/experts/ant_transfer.yml
python common/collect.py configs/samples/experts/ant_transfer.yml
```

After the training is done, you can choose one of the saved reward model to train a policy from scratch (Recovering the Stationary Reward Function).

Transferring the reward to disabled Ant

```bash 
python common/train_optimal.py configs/samples/experts/ant_transfer.yml
python ml/irl_samples.py configs/samples/agents/data_transfer.yml(data transfer)
```
