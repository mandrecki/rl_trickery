import copy
import math
import os
import time
from ast import literal_eval as make_tuple

import numpy as np
from collections import deque

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.backends.cudnn.benchmark = True

import rl_trickery.envs
import rl_trickery.utils.utils as utils
from rl_trickery.utils.logger import Logger
from rl_trickery.utils.video import VideoRecorder
from rl_trickery.envs import make_envs
from rl_trickery.models.tricky_policy_networks import RecursivePolicy, PolicyOutput
from rl_trickery.agents.tricky_agents import A2C, TrickyRollout


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()

        self.cfg = cfg
        # init loggers
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_timestep_interval,
                             agent=cfg.agent.name)

        if self.cfg.seed == 0:
            self.cfg.seed = int(time.time_ns() / 10e9)
        utils.set_seed_everywhere(self.cfg.seed)

        if torch.cuda.is_available():
            self.device = torch.device(self.cfg.device)
        else:
            self.device = "cpu"
            self.cfg.device = self.device

        # init envs
        self.env = make_envs(
            **self.cfg.env,
            num_envs=cfg.num_envs,
            seed=self.cfg.seed
        )

        # init eval envs
        self.eval_envs = make_envs(
            **self.cfg.env,
            num_envs=self.cfg.num_eval_envs,
            seed=self.cfg.seed+1337,
        )

        # init net
        self.net = RecursivePolicy(
            obs_space=self.env.observation_space,
            action_space=self.env.action_space,
            twoAM=self.cfg.agent.algo_params.twoAM,
            **self.cfg.agent.network_params
        )
        self.cfg.model_params_count = utils.get_n_params(self.net)
        print("Model params count:", self.cfg.model_params_count)

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

    def evaluate(self, step):
        eval_episode_rewards = deque(maxlen=self.cfg.num_eval_episodes)
        timeout_cnt = 0.0
        safety_cnt = 0
        obs = self.eval_envs.reset()
        rnn_h = torch.zeros((self.eval_envs.num_envs,) + self.net.recurrent_hidden_state_size()).to(self.device)
        done = torch.zeros((self.eval_envs.num_envs, 1)).to(self.device)
        cog_policy = PolicyOutput(
            value=None, action=torch.ones((self.eval_envs.num_envs, 1), device=self.device).long(),
            action_log_probs=None, dist_entropy=None
        )
        batched_obs_seq = []
        while len(eval_episode_rewards) < self.cfg.num_eval_episodes:
            safety_cnt += 1
            if safety_cnt > 2000:
                break

            if len(self.env.observation_space.shape) == 3:
                batched_obs_seq.append((obs[:,:3].cpu().numpy()).astype("uint8"))

            env_policy, cog_policy, rnn_h = self.net(obs, rnn_h, done, cog_policy.action)
            value, action, _, _ = env_policy

            pause_action = torch.ones_like(action) * self.cfg.env.cognitive_pause
            action = action * cog_policy.action + pause_action * (1 - cog_policy.action)

            obs, _, done, infos = self.eval_envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])
                    if 'TimeLimit.truncated' in info.keys():
                        timeout_cnt += 1

        if len(self.env.observation_space.shape) == 3:
            self.video_recorder.init()
            self.video_recorder.capture(batched_obs_seq)
            self.video_recorder.save(f'{step}.mp4')

        timeout_fraction = timeout_cnt/self.cfg.num_eval_episodes
        return eval_episode_rewards, timeout_fraction

    def run(self):
        timesteps_cnt = 0
        updates_cnt = 0
        episodes_cnt = 0
        episode_rewards = deque(maxlen=50)
        episode_rewards.append(0)
        timesteps_per_update = self.env.num_envs * self.cfg.env.frame_skip * self.cfg.agent.num_steps

        next_obs = self.env.reset()
        rnn_h = torch.zeros((self.env.num_envs,) + self.net.recurrent_hidden_state_size(), device=self.device)
        done = torch.zeros((self.env.num_envs, 1), device=self.device)
        cog_policy = PolicyOutput(
            value=None, action=torch.ones((self.env.num_envs, 1), device=self.device).long(),
            action_log_probs=None, dist_entropy=None
        )

        while timesteps_cnt < self.cfg.num_timesteps:
            start_time = time.time()
            steps_since_update = 0
            safety_cnt = 0

            while steps_since_update < self.cfg.agent.num_steps * self.env.num_envs:
                safety_cnt += 1
                if safety_cnt > 2 * self.cfg.agent.num_steps * self.env.num_envs:
                    break
                obs = next_obs
                env_policy, cog_policy, rnn_h = self.net(obs, rnn_h, done, cog_policy.action)
                value, action, action_logp, action_entropy = env_policy

                pause_action = torch.ones_like(action) * self.cfg.env.cognitive_pause
                action = action * cog_policy.action + pause_action * (1 - cog_policy.action)

                next_obs, reward, done, infos = self.env.step(action)
                timeout = torch.FloatTensor([[1.0] if 'TimeLimit.truncated' in info.keys() else [0.0] for info in infos]).to(self.cfg.device)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                episodes_cnt += done.sum()
                steps_since_update += cog_policy.action.sum()
                self.buffer.append(obs, action, reward, done, timeout, rnn_h, value, action_logp, action_entropy)
                self.buffer.append_cog(*cog_policy)

            timesteps_cnt += steps_since_update * self.cfg.env.frame_skip
            with torch.no_grad():
                env_policy, cog_policy, _ = self.net(obs, rnn_h, done, cog_policy.action)
                self.buffer.v.append(env_policy.value)
                self.buffer.v_c.append(cog_policy.value)

            # value_loss, action_loss, entropy_loss = self.agent.update()  # non-cognitive working update
            env_loss, cog_loss = self.agent.cognitive_update()
            updates_cnt += 1

            if (updates_cnt) % (1+self.cfg.log_timestep_interval//timesteps_per_update) == 0:
                end_time = time.time()
                self.logger.log("train/episode_reward", np.mean(episode_rewards), updates_cnt)
                self.logger.log('train/value', torch.stack(self.buffer.v).mean(), updates_cnt)
                self.logger.log('train/episode', episodes_cnt, updates_cnt)
                self.logger.log('train/timestep', timesteps_cnt, updates_cnt)
                self.logger.log('train/duration', end_time - start_time, updates_cnt)
                self.logger.log('train/fps', timesteps_per_update/(end_time - start_time), updates_cnt)
                self.logger.log('train_loss/critic', env_loss.value, updates_cnt)
                self.logger.log('train_loss/actor', env_loss.action, updates_cnt)
                self.logger.log('train_loss/entropy', env_loss.entropy, updates_cnt)
                if self.agent.twoAM:
                    self.logger.log('train/value_cog', torch.stack(self.buffer.v_c).mean(), updates_cnt)
                    self.logger.log('train/act', torch.stack(self.buffer.a_c).float().mean(), updates_cnt)
                    self.logger.log('train_loss/critic_cog', cog_loss.value, updates_cnt)
                    self.logger.log('train_loss/actor_cog', cog_loss.action, updates_cnt)
                    self.logger.log('train_loss/entropy_cog', cog_loss.entropy, updates_cnt)

                self.logger.dump(updates_cnt)

            if (updates_cnt) % (1+self.cfg.eval_timestep_interval//timesteps_per_update) == 0:
                with torch.no_grad():
                    eval_rewards, eval_fraction_timeouts = self.evaluate(timesteps_cnt)
                self.logger.log("eval/episode_reward", np.mean(eval_rewards), updates_cnt)
                self.logger.log("eval/fraction_timeouts", eval_fraction_timeouts, updates_cnt)
                self.logger.log("eval/timestep", timesteps_cnt, updates_cnt)
                self.logger.dump(updates_cnt)

            self.buffer.after_update()
            rnn_h = rnn_h.detach()

        return np.mean(episode_rewards)


@hydra.main(config_path='configs/', config_name='config')
def main(cfg):
    workspace = Workspace(cfg)
    # print(cfg.pretty())
    mean_reward = workspace.run()
    return float(mean_reward)



if __name__ == '__main__':
    main()
