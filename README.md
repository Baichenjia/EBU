# Reproduce for [Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update](https://arxiv.org/abs/1805.12375) (NeurIPS 2019) with Tensorflow

## Prerequisites
- Tensorflow-gpu > 1.13 with eager execution, or tensorflow 2.x
- Tensorflow-probability 0.6.0
- OpenAI [baselines](https://github.com/openai/baselines)
- OpenAI [Gym](http://gym.openai.com/)

## Usage

The following command should train an agent on "Breakout" for 20M frames.

`python run_atari.py --env BreakoutNoFrameskip-v4`

## Structure Overview

- `deepq.py` contains stepping the environment, storing experience and saving models.
- `deepq_learner.py` contains the action selection and EBU training.
- `replay_buffer.py` contains replay buffer for EBU.
- `models.py` contains Q-network.
- `run_atari.py` contains hyper-parameters setting. Run this file will start training.


## Execution

The data for separate runs is stored on disk under the `result` directory with filename 
`<env-id>-<algorithm>-<date>-<time>.` Each run directory contains
- `log.txt` Record the episode, exploration rate, episodic mean rewards in training 
(after normalization and used for training), episodic mean scores (raw score), current timesteps, percentage completed.
- `monitor.csv` Env monitor file by using `logger` from `Openai Baselines`.
- `parameters.txt` All hyper-parameters used in training.
- `progress.csv` Same data as `log.txt` but with `csv` format.
- `evaluate scores.txt` Evaluation of policy for 108000 frames every 1e5 training steps with 30 no-op evaluation. 
- `model_10M.h5`, `model_20M.h5`, `model_best_10M.h5`, `model_best_20M.h5` are the policy files saved.
