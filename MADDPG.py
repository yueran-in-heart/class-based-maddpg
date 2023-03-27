import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir):
        
        self.agents = {}
        self.buffers = {}
        for i in dim_info['agent_class_name_list']:
            self.agents[i] = Agent(dim_info['self_obs_dim'], dim_info['action_dim'], dim_info['global_obs_act_dim'], actor_lr, critic_lr)
            self.buffers[i] = Buffer(capacity, dim_info['self_obs_dim'], dim_info['action_dim'], 'cpu')

        self.dim_info = dim_info
        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log')) 

    def add(self, obs, action, reward, next_obs, done):
        # NOTE that the experience is a dict with agent name as its key
        for agent_class in obs.keys():
            o = obs[agent_class]
            a = action[agent_class]
            n = next_obs[agent_class]
            # if isinstance(a, int):
            #     # the action from env.action_space.sample() is int, we have to convert it to onehot
            #     a = np.eye(self.dim_info[agent_id][1])[a]
            flag = 0
            for agent_pos in a.keys():
                act = a[agent_pos][0]
                ob = o[flag]
                r = reward
                next_o = n[flag]
                d = done
                self.buffers[agent_class].add(ob, act, r, next_o, d)
                flag +=1

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs):
        actions = {}
        for agent_class, agents_info in obs.items():
            actions[agent_class] = {}
            for o in agents_info:
                pos = o[-2]*16 + o[-1]
                pos_x = o[-2]
                pos_y = o[-1]
                o = torch.from_numpy(o).unsqueeze(0).float()
                a = self.agents[agent_class].action(o)
                actions[agent_class][pos] = a
                self.logger.info(f'class:{agent_class}, pos-x:{pos_x}, pos-y:{pos_y}, action: {a}')
        
        # for agent, o in obs.items():
        #     o = torch.from_numpy(o).unsqueeze(0).float()
        #     a = self.agents[agent].action(o)  # torch.Size([1, action_size])
        #     # NOTE that the output is a tensor, convert it to int before input to the environment
        #     actions[agent] = a.squeeze(0).argmax().item()
        #     self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save(self):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        # with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
        #     pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
