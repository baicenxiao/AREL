**Status:** Archive (code is provided as-is, no updates expected)

## Agent-Temporal Attention for Reward Redistribution in Episodic Multi-Agent Reinforcement Learning (AREL) ##


The repository contains Pytorch implementation of AREL based on MADDPG with Permutation Invariant Critic (PIC).

### Summary
This paper considers multi-agent reinforcement learning (MARL) tasks where agents receive a shared global reward at the end of an episode. The delayed nature of this reward affects the ability of the agents to assess the quality of their actions at intermediate time-steps. This paper focuses on developing methods to learn a temporal redistribution of the episodic reward to obtain a dense reward signal. Solving such MARL problems requires addressing two challenges: identifying (1) relative importance of states along the length of an episode (along time), and (2) relative importance of individual agents’ states at any single time-step (among agents). In this paper, we introduce Agent-Temporal Attention for Reward Redistribution in Episodic Multi-Agent Reinforcement Learning (AREL) to address these two challenges. AREL uses attention mechanisms to characterize the influence of actions on state transitions along trajectories (temporal attention), and how each agent is affected by other agents at each time-step (agent attention). The redistributed rewards predicted by AREL are dense, and can be integrated with any given MARL algorithm.

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

### License
This project is licensed under the MIT License

### Disclaimer

THE SAMPLE CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BAICEN XIAO OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) SUSTAINED BY YOU OR A THIRD PARTY, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT ARISING IN ANY WAY OUT OF THE USE OF THIS SAMPLE CODE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### Acknowledgements
The code of MADDPG with PIC is based on the publicly available implementation of https://github.com/IouJenLiu/PIC

This work was supported by the U.S. Office of Naval Research via Grant N00014-17-S-B001. 

The code of MADDPG is based on the publicly available implementation: https://github.com/openai/maddpg.

### Additional Information

Project Webpage: Feedback-driven Learn to Reason in Adversarial Environments for Autonomic Cyber Systems (http://labs.ece.uw.edu/nsl/faculty/ProjectWebPages/L2RAVE/)


### Paper citation

If you used this code for your experiments or found it helpful, please cite the following paper:

Bibtex:
<pre>
@article{xiao2022arel,
  title={Agent-Temporal Attention for Reward Redistribution in Episodic Multi-Agent Reinforcement Learning
},
  author={Xiao, Baicen and Ramasubramanian, Bhaskar and Poovendran, Radha},
  booktitle={Proceedings of the 21th International Conference on Autonomous Agents and MultiAgent Systems},
  year={2022}
}
</pre>

