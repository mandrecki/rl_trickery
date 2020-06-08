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

# from a2c_ppo_acktr.model import Policy
from rl_trickery.models.policy_networks import Policy, RecurrentPolicy
from a2c_ppo_acktr.algo import PPO, A2C_ACKTR
from a2c_ppo_acktr.storage import RolloutStorage


import rl_trickery.envs
import rl_trickery.utils.utils as utils
from rl_trickery.utils.logger import Logger
from rl_trickery.utils.video import VideoRecorder
from rl_trickery.envs import make_envs
from rl_trickery.data.storage import RolloutStorage

# torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.env = make_envs(
            **cfg.env,
            num_envs=cfg.num_envs,
            seed=self.cfg.seed
        )

        self.eval_envs = make_envs(
            **self.cfg.env,
            num_envs=1,
            seed=self.cfg.seed+1337,
        )
        self.net = RecurrentPolicy(
            self.env.observation_space.shape,
            self.env.action_space,
            **cfg.agent.network,
        )
        self.net.to(self.device)

        if cfg.agent.name == 'a2c':
            self.agent = A2C_ACKTR(
                self.net,
                cfg.agent.value_loss_coef,
                cfg.agent.entropy_coef,
                lr=cfg.agent.lr,
                eps=cfg.agent.eps,
                alpha=cfg.agent.alpha,
                max_grad_norm=cfg.agent.max_grad_norm
            )
        elif cfg.agent.name == 'ppo':
            self.agent = PPO(
                self.net,
                cfg.agent.clip_param,
                cfg.agent.ppo_epoch,
                cfg.agent.num_mini_batch,
                cfg.agent.value_loss_coef,
                cfg.agent.entropy_coef,
                lr=cfg.agent.lr,
                eps=cfg.agent.eps,
                max_grad_norm=cfg.agent.max_grad_norm
            )
        elif cfg.agent.name == 'acktr':
            self.agent = A2C_ACKTR(
                self.net,
                cfg.agent.value_loss_coef,
                cfg.agent.entropy_coef,
                acktr=True
            )

        self.rollouts = RolloutStorage(
            cfg.agent.num_steps,
            cfg.num_envs,
            self.env.observation_space.shape,
            self.env.action_space,
            self.net.recurrent_hidden_state_size
        )
        self.rollouts.to(self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self, step):
        eval_episode_rewards = deque(maxlen=self.cfg.num_eval_episodes)
        obs = self.eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            1,
            *np.atleast_1d(self.net.recurrent_hidden_state_size),
            device=self.device
        )
        eval_masks = torch.zeros(1, 1, device=self.device)

        batched_obs_seq = []
        self.video_recorder.init()
        while len(eval_episode_rewards) < self.cfg.num_eval_episodes:
            batched_obs_seq.append((obs[:,:3].cpu().numpy()).astype("uint8"))
            # self.video_recorder.add_torch_obs(obs)
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = self.net.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    # deterministic=True
                )

            # Obser reward and next obs
            obs, _, done, infos = self.eval_envs.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=self.device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])
        self.video_recorder.capture(batched_obs_seq)
        self.video_recorder.save(f'{step}.mp4')

        return eval_episode_rewards

    def run(self):
        episode_rewards = deque(maxlen=30)
        timesteps_per_update = (self.cfg.agent.num_steps * self.cfg.num_envs * self.cfg.env.frame_skip)
        num_updates = int(self.cfg.num_train_steps // timesteps_per_update)
        total_episodes = 0

        obs = self.env.reset()
        self.rollouts.obs[0].copy_(obs)
        for j in range(num_updates):
            self.step = j
            start_time = time.time()
            for step in range(self.cfg.agent.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.net.act(
                        self.rollouts.obs[step],
                        self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step]
                    )

                obs, reward, done, infos = self.env.step(action)
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]
                )
                # total_episodes += (1 - masks).sum()
                total_episodes += sum(done)

                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos]
                )
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.net.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]
                ).detach()

            self.rollouts.compute_returns(next_value, self.cfg.agent.use_gae, self.cfg.agent.gamma,
                                     self.cfg.agent.gae_lambda, self.cfg.agent.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
            self.rollouts.after_update()

            total_num_steps = (j + 1) * timesteps_per_update
            if j != 0 \
                    and j % (self.cfg.log_frequency_step // self.cfg.agent.num_steps) == 0 \
                    and len(episode_rewards) > 1:
                end_time = time.time()
                self.logger.log("train/episode_reward", np.mean(episode_rewards), total_num_steps)
                # self.logger.log('train/batch_return', self.rollouts.returns.mean(), total_num_steps)
                self.logger.log('train/value', self.rollouts.value_preds.mean(), total_num_steps)
                self.logger.log('train/episode', total_episodes, self.step)
                self.logger.log('train/timestep', total_num_steps, self.step)
                self.logger.log('train/duration', end_time - start_time, self.step)
                self.logger.log('train/fps', timesteps_per_update/(end_time - start_time), self.step)
                self.logger.log('train_loss/critic', value_loss, total_num_steps)
                self.logger.log('train_loss/actor', action_loss, total_num_steps)
                self.logger.log('train_loss/entropy', dist_entropy, total_num_steps)
                self.logger.dump(self.step)

            # if j != 0\
            if j % (self.cfg.eval_frequency_step // self.cfg.agent.num_steps) == 0\
                    or j == num_updates-1:
                eval_rewards = self.evaluate(total_num_steps)
                self.logger.log("eval/episode_reward", np.mean(eval_rewards), total_num_steps)
                self.logger.log('eval/episode', total_episodes, self.step)
                self.logger.log('eval/timestep', total_num_steps, self.step)
                self.logger.dump(self.step)


@hydra.main(config_path='configs/config.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    print(cfg.pretty())
    workspace.run()


if __name__ == '__main__':
    main()
