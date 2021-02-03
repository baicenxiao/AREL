import argparse
import math
from collections import namedtuple
from itertools import count
import numpy as np
from eval import eval_model_q
import copy
import torch
from ddpg_vec import DDPG
from ddpg_vec_hetero import DDPGH
import random
import pickle

from replay_memory import ReplayMemory, Transition, ReplayMemory_episode
from utils import *
import os
import time
from utils import n_actions, copy_actor_policy
from ddpg_vec import hard_update
import torch.multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value
import sys
from pathlib import Path

save_path = str(Path(os.path.abspath(__file__)).parents[2]) + '/results'
save_model_path = save_path + '/ckpt_plot'
tensorboard_path = save_path + '/runs'

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1])+'/former')
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1])+'/maddpg')
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

import numpy as np
import torch
from former import util
from former.util import d, here
import former

from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter





parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--scenario', required=True,
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--train_noise', default=False, action='store_true')
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=60000, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=25, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=100000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=8, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--critic_updates_per_step', type=int, default=8, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--actor_lr', type=float, default=1e-2,
                    help='(default: 1e-4)')
parser.add_argument('--critic_lr', type=float, default=1e-2,
                    help='(default: 1e-3)')
parser.add_argument('--fixed_lr', default=False, action='store_true')
parser.add_argument('--num_eval_runs', type=int, default=1000, help='number of runs per evaluation (default: 5)')
parser.add_argument("--exp_name", type=str, help="name of the experiment")
parser.add_argument("--save_dir", type=str, default=save_model_path,
                    help="directory in which training state and model should be saved")
parser.add_argument('--static_env', default=False, action='store_true')
parser.add_argument('--critic_type', type=str, default='mlp', help="Supports [mlp, gcn_mean, gcn_max]")
parser.add_argument('--actor_type', type=str, default='mlp', help="Supports [mlp, gcn_max]")
parser.add_argument('--critic_dec_cen', default='cen')
parser.add_argument("--env_agent_ckpt", type=str, default='ckpt_plot/simple_tag_v5_al0a10_4/agents.ckpt')
parser.add_argument('--shuffle', default=None, type=str, help='None|shuffle|sort')
parser.add_argument('--episode_per_update', type=int, default=4, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--episode_per_actor_update', type=int, default=4)
parser.add_argument('--episode_per_critic_update', type=int, default=4)
parser.add_argument('--steps_per_actor_update', type=int, default=100)
parser.add_argument('--steps_per_critic_update', type=int, default=100)
#parser.add_argument('--episodes_per_update', type=int, default=4)
parser.add_argument('--target_update_mode', default='soft', help='soft | hard | episodic')
parser.add_argument('--cuda', default=False, action='store_true')
parser.add_argument('--eval_freq', type=int, default=1000)
args = parser.parse_args()
if args.exp_name is None:
    args.exp_name = args.scenario + '_' + args.critic_type + '_' + args.target_update_mode + '_hiddensize' \
                    + str(args.hidden_size) + '_' + str(args.seed)
print("=================Arguments==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")


torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


env = make_env(args.scenario, None)
n_agents = env.n
env.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
num_adversary = 0


n_actions = n_actions(env.action_space)
obs_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
obs_dims.insert(0, 0)
if 'hetero' in args.scenario:
    import multiagent.scenarios as scenarios
    groups = scenarios.load(args.scenario + ".py").Scenario().group
    agent = DDPGH(args.gamma, args.tau, args.hidden_size,
                 env.observation_space[0].shape[0], n_actions[0], n_agents, obs_dims, 0,
                 args.actor_lr, args.critic_lr,
                 args.fixed_lr, args.critic_type, args.train_noise, args.num_episodes,
                 args.num_steps, args.critic_dec_cen, args.target_update_mode, device, groups=groups)
    eval_agent = DDPGH(args.gamma, args.tau, args.hidden_size,
                 env.observation_space[0].shape[0], n_actions[0], n_agents, obs_dims, 0,
                 args.actor_lr, args.critic_lr,
                 args.fixed_lr, args.critic_type, args.train_noise, args.num_episodes,
                 args.num_steps, args.critic_dec_cen, args.target_update_mode, 'cpu', groups=groups)
else:
    agent = DDPG(args.gamma, args.tau, args.hidden_size,
                 env.observation_space[0].shape[0], n_actions[0], n_agents, obs_dims, 0,
                 args.actor_lr, args.critic_lr,
                 args.fixed_lr, args.critic_type, args.actor_type, args.train_noise, args.num_episodes,
                 args.num_steps, args.critic_dec_cen, args.target_update_mode, device)
    eval_agent = DDPG(args.gamma, args.tau, args.hidden_size,
                 env.observation_space[0].shape[0], n_actions[0], n_agents, obs_dims, 0,
                 args.actor_lr, args.critic_lr,
                 args.fixed_lr, args.critic_type, args.actor_type, args.train_noise, args.num_episodes,
                 args.num_steps, args.critic_dec_cen, args.target_update_mode, 'cpu')
memory = ReplayMemory(args.replay_size)
memory_e = ReplayMemory_episode(int(args.replay_size/args.num_steps))
feat_dims = []
for i in range(n_agents):
    feat_dims.append(env.observation_space[i].shape[0])

# Find main agents index
unique_dims = list(set(feat_dims))
agents0 = [i for i, feat_dim in enumerate(feat_dims) if feat_dim == unique_dims[0]]
if len(unique_dims) > 1:
    agents1 = [i for i, feat_dim in enumerate(feat_dims) if feat_dim == unique_dims[1]]
    main_agents = agents0 if len(agents0) >= len(agents1) else agents1
else:
    main_agents = agents0

rewards = []
total_numsteps = 0
updates = 0
exp_save_dir = os.path.join(args.save_dir, args.exp_name)
os.makedirs(exp_save_dir, exist_ok=True)
best_eval_reward, best_good_eval_reward, best_adversary_eval_reward = -1000000000, -1000000000, -1000000000
start_time = time.time()
value_loss, policy_loss = 0.0, 0.0
copy_actor_policy(agent, eval_agent)
torch.save({'agents': eval_agent}, os.path.join(exp_save_dir, 'agents_best.ckpt'))
# for mp test

test_q = Queue()
done_training = Value('i', False)
p = mp.Process(target=eval_model_q, args=(test_q, done_training, args))
p.start()

####################################################
Transition_e = namedtuple(
    'Transition_episode', ('states', 'actions', 'masks', 'next_states', 'rewards'))

use_agent_attention = False
use_rudder = False

def sample_trajectory(memory, n_trajectories=128, use_agent_attention=False, use_rudder=False):
    sample_traj = random.sample(memory, n_trajectories)
    samples = Transition_e(*zip(*sample_traj))

    next_states = np.array(list(samples.next_states)).reshape(n_trajectories, args.num_steps, n_agents, -1).transpose((0,2,1,3))
    states = np.array(list(samples.states)).reshape(n_trajectories, args.num_steps, n_agents, -1).transpose((0,2,1,3))
    all_states = np.concatenate((states[:,:,0,:][:,:,np.newaxis,:], next_states), axis=2)
    rewards = np.array(list(samples.rewards)).reshape(n_trajectories, args.num_steps, n_agents, -1).transpose((0,2,1,3))

    episode_reward = np.sum(np.squeeze(rewards[:,0,:,:],axis=2), axis=1)
    episode_time_reward = np.squeeze(rewards[:,0,:,:],axis=2)
    
    x_train_tensor = torch.from_numpy(next_states).float().contiguous()
    xall_train_tensor = torch.from_numpy(all_states).float().contiguous()
    y_train_tensor = torch.from_numpy(episode_reward).float().contiguous()
    episode_time_reward_tensor = torch.from_numpy(episode_time_reward).float().contiguous()
    
    if use_agent_attention:
        return x_train_tensor, y_train_tensor, episode_time_reward_tensor
    elif use_rudder:
        return xall_train_tensor.transpose(1,2).contiguous().view(n_trajectories, args.num_steps+1, -1), y_train_tensor, episode_time_reward_tensor
    else:
        return x_train_tensor.transpose(1,2).contiguous().view(n_trajectories, args.num_steps, -1), y_train_tensor, episode_time_reward_tensor

def sample_and_pred(memory, model, batch_size, n_agents, n_trajectories=128, 
                            use_agent_attention=False, use_rudder=False):
    sample_traj = random.sample(memory, n_trajectories)
    samples = Transition_e(*zip(*sample_traj))

    next_states = np.array(list(samples.next_states)).reshape(n_trajectories, args.num_steps, n_agents, -1).transpose((0,2,1,3))
    # Construct inputs
    if not use_agent_attention:
        if use_rudder:
            states = np.array(list(samples.states)).reshape(n_trajectories, args.num_steps, n_agents, -1).transpose((0,2,1,3))
            all_states = np.concatenate((states[:,:,0,:][:,:,np.newaxis,:], next_states), axis=2)
            x_train_tensor = torch.from_numpy(all_states).float().transpose(1,2).contiguous().view(n_trajectories, args.num_steps+1, -1)
            hidden = model.init_hidden(n_trajectories)
        else:
            x_train_tensor = torch.from_numpy(next_states).float().transpose(1,2).contiguous().view(n_trajectories, args.num_steps, -1)
    else:
        x_train_tensor = torch.from_numpy(next_states).float().contiguous()

    # Predit rewards
    if use_rudder:
        _, y_time_hat = model(x_train_tensor.to(device), hidden)
    else:
        _, y_time_hat = model(x_train_tensor.to(device))

    pred_rewards = np.repeat(y_time_hat.detach().cpu().numpy()[:,:,np.newaxis], n_agents, axis=2)
    
    # Randomly sample (s,a,r,s')
    ind1 = np.arange(n_trajectories)
    ind2 = np.random.randint(args.num_steps, size=(n_trajectories, int(batch_size/n_trajectories)))
    
    states = np.array(samples.states)[ind1[:,None], ind2].reshape(batch_size,-1)
    actions = np.array(samples.actions)[ind1[:,None], ind2].reshape(batch_size,-1)
    masks = np.array(samples.masks)[ind1[:,None], ind2].reshape(batch_size,-1)
    next_states = np.array(samples.next_states)[ind1[:,None], ind2].reshape(batch_size,-1)
    rewards = pred_rewards[ind1[:,None], ind2].reshape(batch_size,-1)
    
    return Transition(states, actions, masks, next_states, rewards)

class LstmPredNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, drop_prob=0.0):
        super(LstmPredNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden):
        batch_size, t, e = x.size()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
                
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        out = out.view(batch_size, -1)
        final = out[:,-1]
        step_reward = out[:,1:] - out[:,0:-1]
        return final, step_reward
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
        
time_length, state_emb = args.num_steps, obs_dims[-1]

n_heads = 6
depth = 3

if use_agent_attention:
    reward_model = former.Time_Agent_Transformer(emb=state_emb, heads=n_heads, 
                            depth=depth, seq_length=time_length, 
                            n_agents=n_agents, agent=True, dropout=0.0)
elif use_rudder:
    reward_model = LstmPredNet(embedding_dim=n_agents*state_emb, hidden_dim=512, n_layers=2)
else:
    reward_model = former.Time_Transformer(emb=n_agents*state_emb, heads=n_heads, 
                            depth=depth, seq_length=time_length, 
                            n_agents=n_agents, dropout=0.0)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    reward_model = nn.DataParallel(reward_model)

if torch.cuda.is_available():
    reward_model.cuda()

# model(x.to(device))
opt = torch.optim.Adam(lr=0.0001, params=reward_model.parameters(), weight_decay=1e-5)
loss_fn = nn.MSELoss(reduction='mean')

def make_train_step(model, loss_fn, optimizer, use_rudder=False):
    # Builds function that performs a step in the train loop
    def train_step(x, y, z):
        # Sets model to TRAIN mode
        model.train()
        if use_rudder:
            batch_size, _, _ = x.size()
            hidden = model.init_hidden(batch_size)
            yhat, y_time_hat = model(x, hidden)
        # Makes predictions
        else:
            yhat, y_time_hat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        loss_2 = loss_fn(z, y_time_hat)
        
        loss_total = loss + 20*torch.mean(torch.var(y_time_hat, dim=1))
        # Computes gradients
        loss_total.backward()
        # loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item(), loss_2.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

# Creates the train_step function for our model, loss function and optimizer
train_step = make_train_step(reward_model, loss_fn, opt, use_rudder=use_rudder)

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(tensorboard_path)
####################################################

for i_episode in range(args.num_episodes):
    obs_n = env.reset()
    episode_reward = 0
    episode_step = 0
    agents_rew = [[] for _ in range(n_agents)]
    x_e, action_e, mask_e, x_next_e, reward_e = [], [], [], [], []
    while True:
        # action_n_1 = [agent.select_action(torch.Tensor([obs]).to(device), action_noise=True, param_noise=False).squeeze().cpu().numpy() for obs in obs_n]
        action_n = agent.select_action(torch.Tensor(obs_n).to(device), action_noise=True,
                                       param_noise=False).squeeze().cpu().numpy()
        next_obs_n, reward_n, done_n, info = env.step(action_n)
        total_numsteps += 1
        episode_step += 1
        terminal = (episode_step >= args.num_steps)

        action = torch.Tensor(action_n).view(1, -1)
        mask = torch.Tensor([[not done for done in done_n]])
        next_x = torch.Tensor(np.concatenate(next_obs_n, axis=0)).view(1, -1)

        episode_reward += np.sum(reward_n)
        # if done_n[0] or terminal:
        #     reward = torch.Tensor([[episode_reward/(n_agents*25)]*n_agents])
        # else:
        #     reward = torch.Tensor([[0.0]*n_agents])
        # reward = torch.Tensor([reward_n])
        
        # x = torch.Tensor(np.concatenate(obs_n, axis=0)).view(1, -1)
        # memory.push(x, action, mask, next_x, reward)

        x_e.append(np.concatenate(obs_n, axis=0).reshape(1,-1))
        action_e.append(action_n.reshape(1,-1))
        mask_e.append(np.array([[not done for done in done_n]]))
        x_next_e.append(np.concatenate(next_obs_n, axis=0).reshape(1,-1))
        reward_e.append(np.array([reward_n]))
        # x_e.append(torch.Tensor(np.concatenate(obs_n, axis=0)).view(1, -1))
        # action_e.append(torch.Tensor(action_n).view(1, -1))
        # mask_e.append(torch.Tensor([[not done for done in done_n]]))
        # x_next_e.append(torch.Tensor(np.concatenate(next_obs_n, axis=0)).view(1, -1))
        # reward_e.append(torch.Tensor([reward_n]))
        # print(obs_n[0].shape, len(obs_n))

        for i, r in enumerate(reward_n):
            agents_rew[i].append(r)
        
        obs_n = next_obs_n
        n_update_iter = 5
        # if len(memory) > args.batch_size:
        if len(memory_e) > args.batch_size * 5:
        # if len(memory_e)*4 >= args.batch_size:
            ################################################

            if total_numsteps % args.steps_per_actor_update == 0:
                for _ in range(args.updates_per_step):
                    # transitions = memory.sample(args.batch_size)
                    # batch = Transition(*zip(*transitions))
                    # batch = memory_e.sample(args.batch_size)
                    batch = sample_and_pred(memory_e.memory, reward_model, 
                                    args.batch_size, n_agents, 
                                    n_trajectories=256, use_agent_attention=use_agent_attention,
                                    use_rudder= use_rudder)
                    # print(batch.reward.shape)
                    
                    policy_loss = agent.update_actor_parameters(batch, i, args.shuffle)
                    updates += 1
                print('episode {}, p loss {}, p_lr {}'.
                      format(i_episode, policy_loss, agent.actor_lr))
            if total_numsteps % args.steps_per_critic_update == 0:
                value_losses = []
                for _ in range(args.critic_updates_per_step):
                    # transitions = memory.sample(args.batch_size)
                    # batch = Transition(*zip(*transitions))
                    # batch = memory_e.sample(args.batch_size)
                    batch = sample_and_pred(memory_e.memory, reward_model, 
                                    args.batch_size, n_agents, 
                                    n_trajectories=256, use_agent_attention=use_agent_attention,
                                    use_rudder= use_rudder)
                    
                    value_losses.append(agent.update_critic_parameters(batch, i, args.shuffle)[0])
                    updates += 1
                    # print(value_losses)
                value_loss = np.mean(value_losses)
                print('episode {}, q loss {},  q_lr {}'.
                      format(i_episode, value_loss, agent.critic_optim.param_groups[0]['lr']))
                if args.target_update_mode == 'episodic':
                    hard_update(agent.critic_target, agent.critic)

        if done_n[0] or terminal:
            print('train epidoe reward', episode_reward, ' episode ', i_episode)
            episode_step = 0
            memory_e.push(x_e, action_e, mask_e, x_next_e, reward_e)
            x_e, action_e, mask_e, x_next_e, reward_e = [], [], [], [], []
            break
    # print(len(memory_e))
    # if len(memory_e)==12000:
    #         with open('trajectories3.pickle', 'wb') as handle:
    #             pickle.dump(memory_e, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ################################################
    # train the reward redistribution model
        
    # if i_episode % 1000 == 0 or (len(memory_e) == args.batch_size * 5):
    if (i_episode+1) % 1000 == 0 and (len(memory_e)>4000):
        epoch_train_episode_reward_loss = []
        epoch_train_step_reward_loss = []
        for ii in range(1000):
            x_batch, y_batch, z_batch = sample_trajectory(memory_e.memory, n_trajectories=128, 
                                                    use_agent_attention=use_agent_attention,
                                                    use_rudder=use_rudder)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch = z_batch.to(device)
            loss, loss_2 = train_step(x_batch, y_batch, z_batch)
            epoch_train_episode_reward_loss.append(loss)
            epoch_train_step_reward_loss.append(loss_2)
        writer.add_scalar(args.exp_name + f'_episode_reward_loss', np.mean(epoch_train_episode_reward_loss), i_episode)
        writer.add_scalar(args.exp_name + f'_step_reward_loss', np.mean(epoch_train_step_reward_loss), i_episode)

    if not args.fixed_lr:
        agent.adjust_lr(i_episode)
    writer.add_scalar(args.exp_name + f'_episode_reward', episode_reward, i_episode)
    rewards.append(episode_reward)
    # if (i_episode + 1) % 1000 == 0 or ((i_episode + 1) >= args.num_episodes - 50 and (i_episode + 1) % 4 == 0):
    if (i_episode + 1) % args.eval_freq == 0:
        tr_log = {'num_adversary': 0,
                  'best_good_eval_reward': best_good_eval_reward,
                  'best_adversary_eval_reward': best_adversary_eval_reward,
                  'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
                  'value_loss': value_loss, 'policy_loss': policy_loss,
                  'i_episode': i_episode, 'start_time': start_time}
        copy_actor_policy(agent, eval_agent)
        test_q.put([eval_agent, tr_log])

env.close()
time.sleep(5)
done_training.value = True
