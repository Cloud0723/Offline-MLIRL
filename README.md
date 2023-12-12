# When Demonstrations meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning
Offline ML-IRL is an algorithm for offline inverse reinforcement learning that is discussed in the article [arxiv link](http://arxiv.org/abs/2302.07457)

## Installation
- PyTorch 1.13.1
- MuJoCo 2.1.0
- pip install -r requirements.txt


## File Structure
- Experiment result ：`data/`
- Configurations: `args_yml/`
- Expert Demonstrations: `expert_data/`

## Instructions
- All the experiments are to be run under the root folder. 
- After running, you will see the training logs in `data/` folder.

## Experiments
All the commands below are also provided in `run.sh`.

### Offline-IRL benchmark (MuJoCo)
Before experiment, you can download our expert demonstrations and our trained world model [here](https://drive.google.com/drive/folders/1BbEZLEKP6HAijeRBXG0V3JLSrB0FIQg6?usp=drive_link).

```bash
python train.py --yaml_file args_yml/model_base_IRL/halfcheetah_v2_medium.yml --seed 0 --uuid halfcheetah_result 
```


## Performances
![Graph](imgs/fig_1.png)

-----

![Graph](imgs/fig_2.png)

----
