import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np
from collections import deque

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

import rl_trickery.envs
import rl_trickery.utils.utils as utils
from rl_trickery.utils.logger import Logger
from rl_trickery.utils.video import VideoRecorder
from rl_trickery.envs import make_envs
from rl_trickery.models.tricky_policy_networks import RecursivePolicy
from rl_trickery.agents.tricky_agents import A2C, TrickyRollout


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        # init loggers
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # init envs
        self.env = make_envs(
            **cfg.env,
            num_envs=cfg.num_envs,
            seed=self.cfg.seed
        )

        # init eval envs
        self.eval_envs = make_envs(
            **self.cfg.env,
            num_envs=2,
            seed=self.cfg.seed+1337,
        )

        # init net
        self.net = RecursivePolicy(
            obs_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.cfg.agent.network_params
        )
        print("Model params count:", utils.get_n_params(self.net))
        self.net.to(self.device)

        # init storage
        self.buffer = TrickyRollout()

        # init agent
        if cfg.agent.name == 'a2c':
            self.agent = A2C(
                self.net,
                self.buffer,
                **self.cfg.agent.algo_params,
            )
        else:
            raise NotImplementedError

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)

    def run(self):
        timesteps_cnt = 0
        updates_cnt = 0
        episodes_cnt = 0
        episode_rewards = deque(maxlen=30)
        episode_rewards.append(0)

        next_obs = self.env.reset()
        rnn_h = torch.zeros((self.env.num_envs,) + self.net.recurrent_hidden_state_size()).to(self.device)
        done = torch.zeros((self.env.num_envs, 1)).to(self.device)

        while timesteps_cnt < self.cfg.num_timesteps:
            start_time = time.time()
            for t in range(self.cfg.agent.num_steps):
                obs = next_obs
                value, action, action_logp, action_entropy, rnn_h = self.net.act(obs, rnn_h, done)

                next_obs, reward, done, infos = self.env.step(action)
                # TODO change bad transition extraction
                timeout = torch.FloatTensor([[1.0] if 'TimeLimit.truncated' in info.keys() else [0.0] for info in infos]).to(self.cfg.device)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                episodes_cnt += done.sum()
                timesteps_cnt += self.env.num_envs * self.cfg.env.frame_skip
                self.buffer.append(obs, action, reward, done, timeout, rnn_h, value, action_logp, action_entropy)

            with torch.no_grad():
                value = self.net.get_value(next_obs, rnn_h, done)
                self.buffer.v.append(value)

            value_loss, action_loss, entropy_loss = self.agent.update()
            updates_cnt += 1

            if updates_cnt % self.cfg.log_frequency_step == 0:
                end_time = time.time()
                timesteps_per_update = self.env.num_envs * self.cfg.env.frame_skip * self.cfg.agent.num_steps
                self.logger.log("train/episode_reward", np.mean(episode_rewards), updates_cnt)
                self.logger.log('train/value', torch.stack(self.buffer.v).mean(), updates_cnt)
                self.logger.log('train/episode', episodes_cnt, updates_cnt)
                self.logger.log('train/timestep', timesteps_cnt, updates_cnt)
                self.logger.log('train/duration', end_time - start_time, updates_cnt)
                self.logger.log('train/fps', timesteps_per_update/(end_time - start_time), updates_cnt)
                self.logger.log('train_loss/critic', value_loss, updates_cnt)
                self.logger.log('train_loss/actor', action_loss, updates_cnt)
                self.logger.log('train_loss/entropy', entropy_loss, updates_cnt)
                self.logger.dump(updates_cnt)

            # if torch.stack(self.buffer.done).sum() > 0:
            #     print("now!")
            # if torch.stack(self.buffer.timeout).sum() > 0:
            #     print("now2!")
            self.buffer.after_update()
            rnn_h = rnn_h.detach()


@hydra.main(config_path='configs/', config_name='simple')
def main(cfg):
    workspace = Workspace(cfg)
    print(cfg.pretty())
    workspace.run()


if __name__ == '__main__':
    main()
