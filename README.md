## Agent-Temporal Attention for Reward Redistribution in Episodic Multi-Agent Reinforcement Learning (AREL) ##


The repository contains Pytorch implementation of AREL based on MADDPG with Permutation Invariant Critic (PIC).

#### Platform and Dependencies: 
* Ubuntu 18.04 
* Python 3.7
* Pytorch 1.6.0
* OpenAI gym 0.10.9 (https://github.com/openai/gym)

### Install the improved MPE:
    cd multiagent-particle-envs
    pip install -e .
Please ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH`.

## Training examples
The following are sample commands using different credit assignment methods for MARL training in the Predator-Prey environment with `15 predators`.

### Agent-temporal attention (AREL)
	python maddpg/main_vec_dist_AREL.py --exp_name simple_tag_AREL_n15 --scenario simple_tag_n15 --num_steps=50 --num_episodes=100000 --critic_type gcn_max --cuda
### RUDDER 
	python maddpg/main_vec_dist_RUDDER.py --exp_name simple_tag_RUDDER_n15 --scenario simple_tag_n15 --num_steps=50 --num_episodes=100000 --critic_type gcn_max --cuda
### Trajectory-space smoothing (IRCR)
	python maddpg/main_vec_dist_IRCR.py --exp_name simple_tag_smooth_n15 --scenario simple_tag_n15 --num_steps=50 --num_episodes=100000 --critic_type gcn_max --cuda
### Sequence modeling
	python maddpg/main_vec_dist_SeqMod.py --exp_name simple_tag_TimeAtt_n15 --scenario simple_tag_n15 --num_steps=50 --num_episodes=100000 --critic_type gcn_max --cuda

Results will be saved in `results` folder in the parent directory.

### Acknowledgement
The code of MADDPG with PIC is based on the publicly available implementation of https://github.com/IouJenLiu/PIC

### License
This project is licensed under the MIT License

