import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from MADDPG import MADDPG

import numpy as np

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import (
    MicroRTSGridModeSharedMemVecEnv as MicroRTSGridModeVecEnv,
)
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
import time
import torch
from gym.spaces import MultiDiscrete
from distutils.util import strtobool



class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos[:])
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i])
                * np.concatenate((infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()), axis=None)
            ]
            self.ts[i] += 1
            if dones[i]:
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(0)
                raw_discount_names = ["discounted_" + str(rf) for rf in self.rfs] + ["discounted"]
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(dict(zip(raw_discount_names, raw_discount_returns)))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos

def write_dict(target,key1,key2,value):
    target[key1][key2].append(value)
    return target

def obs_process(obs):
    num_env,w,h,n = obs.shape
    all_envs_info = []
    resource_info = []
    for i in range(num_env):
        agent_infos = {"my_info":{},"opp_info":{}}
        for agent_type in ["base","barrack","worker","light","heavy","ranged"]:
            agent_infos["my_info"][agent_type] = []  
            agent_infos["opp_info"][agent_type] = []
        env_res_info = {"my_info":{},"opp_info":{}}
        for agent_type in ["resource"]:
            env_res_info["my_info"][agent_type] = []  
            env_res_info["opp_info"][agent_type] = []  
        for j in range(w):
            for k in range(h):
                if obs[i][j][k][13]!=1:
                    inf = obs[i][j][k]
                    inf_all = np.append(inf, np.array([j,k]))
                    if inf[14]==1:
                        env_res_info["my_info"]["resource"].append(inf_all)
                        env_res_info["opp_info"]["resource"].append(inf_all)
                    if inf[15]==1:
                        if inf[11] == 1:
                            agent_infos = write_dict(agent_infos,"my_info","base",inf_all)
                        elif inf[12] == 1:
                            agent_infos = write_dict(agent_infos,"opp_info","base",inf_all)
                    if inf[16]==1:
                        if inf[11] == 1:
                            agent_infos = write_dict(agent_infos,"my_info","barrack",inf_all)
                        elif inf[12] == 1:
                            agent_infos = write_dict(agent_infos,"opp_info","barrack",inf_all)
                    if inf[17]==1:
                        if inf[11] == 1:
                            agent_infos = write_dict(agent_infos,"my_info","worker",inf_all)
                        elif inf[12] == 1:
                            agent_infos = write_dict(agent_infos,"opp_info","worker",inf_all)
                    if inf[18]==1:
                        if inf[11] == 1:
                            agent_infos = write_dict(agent_infos,"my_info","light",inf_all)
                        elif inf[12] == 1:
                            agent_infos = write_dict(agent_infos,"opp_info","light",inf_all)
                    if inf[19]==1:
                        if inf[11] == 1:
                            agent_infos = write_dict(agent_infos,"my_info","heavy",inf_all)
                        elif inf[12] == 1:
                            agent_infos = write_dict(agent_infos,"opp_info","heavy",inf_all)
                    if inf[20]==1:
                        if inf[11] == 1:
                            agent_infos = write_dict(agent_infos,"my_info","ranged",inf_all)
                        elif inf[12] == 1:
                            agent_infos = write_dict(agent_infos,"opp_info","ranged",inf_all)
        all_envs_info.append(agent_infos)
        resource_info.append(env_res_info)
    return all_envs_info, resource_info



def get_env(args):
    """create environment and get observation and action dimension of each agent in this environment"""
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(1)],
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    individual_obs,resource_info = obs_process(envs.reset())
    # indivadual_obs: {"my_info":
                            # {"resource":[array(29),array(29),array(29),array(29)],
                            # "base":[],
                            # "barrack":[],
                            # "worker":[],
                            # "light":[],
                            # "heavy":[],
                            # "renged":[]},
                    # "opp_info":# {"resource":[array(29),array(29),array(29),array(29)],
                            # "base":[],
                            # "barrack":[],
                            # "worker":[],
                            # "light":[],
                            # "heavy":[],
                            # "renged":[]}
                    # }

    envs.reset()
    _dim_info = {}
    _dim_info["map_size"] = 16 
    _dim_info["agent_class_num"] = 6 
    _dim_info['agent_class_name_list'] = ['base','barrack','worker',
    'light','heavy','renged']
    #需要做action的agent有六种：基地、营房、工人、轻型攻击者、重型攻击者、远程攻击者
    _dim_info["self_obs_dim"] = 29 # 某个类型的单位自己的obs信息维度：27+位置的二维坐标
    _dim_info["action_dim"] = 7 #每个agent的动作空间是7
    _dim_info["global_obs_act_dim"] = 16*16*27
    return envs, _dim_info

def action_wrapper(dim_info, action):
    action_map = torch.zeros([1,dim_info['map_size']*dim_info['map_size'],dim_info['action_dim']])
    for agent_class, agents_act in action.items():
        for pos, act in agents_act.items():
            action_map[0][pos] = act
    return action_map







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
    #                     choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=1,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
        # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    
    parser.add_argument('--train-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during training')
    args = parser.parse_args()
    
    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # create folder to save result
    env_dir = os.path.join('./results', args.exp_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir)


    for episode in range(args.episode_num):
        obs = env.reset()
        # env.render()
        individual_obs ,resource_info = obs_process(obs)



        # while env.agents:  # interact with the env for an episode
        for step in range(0, args.num_steps):
            env.render()
            
            if step < args.random_steps or step%10!=0:
                action= np.zeros(shape=(1,1792),dtype=np.int64)
                # action = torch.Tensor(action).to(device)
                next_obs, reward, done, infos = env.step(action)
            else:
                action = maddpg.select_action(individual_obs[0]['my_info'])
                # action['worker'][17][0][0] = 1
                # action['worker'][17][0][1] = 1
                a = action_wrapper(dim_info, action) # (1,256,7)
                a = a.detach().numpy().reshape(env.num_envs, -1) #(1,1792)
                b = a.astype(int)

                next_obs, reward, done, infos = env.step(b)
            
            

            if (step < args.random_steps or step%10!=0) is False:
                maddpg.add(obs, b, reward, next_obs, done)

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                # maddpg.learn(args.batch_size, args.gamma)
                # maddpg.update_target(args.tau)
                a=0
            
            next_individual_obs ,next_resource_info = obs_process(next_obs)
            individual_obs = next_individual_obs
            obs = next_obs

    maddpg.save()  # save model
