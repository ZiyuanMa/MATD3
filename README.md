# MATD3
An implementation of Multi-agent TD3.

## Experiment Environment
A simple multi-agent particle world based on gym. Please see [here](https://github.com/openai/multiagent-particle-envs) to install and know more about the environment.


## How to use
### Dependencies:
+ python3.5+
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ [parl>=2.0.4](https://github.com/PaddlePaddle/PARL)
+ PettingZoo==1.17.0
+ gym==0.23.1


### Start Training:
```
# To train an agent for simple_speaker_listener scenario
python train.py

# To train for other scenario, model is automatically saved every 1000 episodes
python train.py --env [ENV_NAME]

# To show animation effects after training
python train.py --env [ENV_NAME] --show --restore

# To train and evaluate scenarios with continuous action spaces
python train.py --env [ENV_NAME] --continuous_actions
python train.py --env [ENV_NAME] --continuous_actions --show --restore
```
