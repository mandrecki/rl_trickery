import os
import time

import numpy as np
from collections import deque

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader


torch.backends.cudnn.benchmark = True

from rl_trickery.envs.maze import MazelabEnv
from rl_trickery.envs.wrappers import ResizeImage, TransposeImage
from rl_trickery.data.maze_storage import generate_dataset
from rl_trickery.models.supervised_maze import MazeSolver
import rl_trickery.utils.utils as utils
from rl_trickery.utils.logger import Logger


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()

        self.cfg = cfg
        # init loggers
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_timestep_interval,
                             agent="supervised_maze")

        if self.cfg.seed == 0:
            self.cfg.seed = int(time.time_ns() / 10e9)
        utils.set_seed_everywhere(self.cfg.seed)

        if torch.cuda.is_available():
            self.device = torch.device(self.cfg.device)
        else:
            self.device = "cpu"
            self.cfg.device = self.device

        # init envs
        env = MazelabEnv(maze_size=self.cfg.maze_size, maze_kind="maze", goal_fixed=False, maze_fixed=False, goal_reward=False,
                         wall_reward=False)
        env = ResizeImage(env, (64, 64), antialias=True)
        env = TransposeImage(env)

        self.dl_train = DataLoader(
            generate_dataset(env, self.cfg.train_mazes),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True
        )
        self.dl_eval = DataLoader(
            generate_dataset(env, self.cfg.eval_mazes),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True
        )

        # init net
        self.net = MazeSolver(
            env.observation_space,
            **self.cfg.network
        )
        self.cfg.model_params_count = utils.get_n_params(self.net)
        print("Model params count:", self.cfg.model_params_count)

        self.net.to(self.device)

    def evaluate(self):
        test_loss = 0
        rnn_h = torch.zeros((self.cfg.batch_size,) + self.net.recurrent_hidden_state_size()).to(self.device)
        done = torch.ones((self.cfg.batch_size, 1)).to(self.device)
        for i_batch, data in enumerate(self.dl_eval):
            x, y = data
            y_hat, _ = self.net(x.cuda(), rnn_h, done)
            test_loss += F.smooth_l1_loss(y_hat.cpu(), y)

        test_loss /= i_batch + 1
        return test_loss

    def run(self):
        rnn_h = torch.zeros((self.cfg.batch_size,) + self.net.recurrent_hidden_state_size()).to(self.device)
        done = torch.ones((self.cfg.batch_size, 1)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        updates_cnt = 0
        for i_epoch in range(self.cfg.epochs):
            start_time = time.time()
            total_loss = 0
            for i_batch, data in enumerate(self.dl_train):
                self.net.zero_grad()

                x, y = data
                y = y.cuda()
                x = x.cuda()

                y_hat, rnn_h = self.net(x, rnn_h.detach(), done)
                loss = F.smooth_l1_loss(y_hat, y)
                loss.backward()
                opt.step()
                total_loss += loss
                updates_cnt += 1

            total_loss /= i_batch + 1
            # if updates_cnt % self.cfg.log_timestep_interval == 0:
            end_time = time.time()
            self.logger.log("train/episode_reward", total_loss, i_epoch)
            self.logger.log("train/out_var", y_hat.var()/y.var(), i_epoch)
            self.logger.log('train/duration', end_time - start_time, i_epoch)
            self.logger.log('train/fps', self.cfg.train_mazes/(end_time - start_time), i_epoch)
            self.logger.log('train/timestep', updates_cnt, i_epoch)
            self.logger.dump(i_epoch)

            # if updates_cnt % self.cfg.eval_timestep_interval == 0:
            with torch.no_grad():
                eval_loss = self.evaluate()
            self.logger.log("eval/episode_reward", eval_loss, i_epoch)
            self.logger.log('eval/timestep', updates_cnt, i_epoch)
            self.logger.dump(i_epoch)

        return eval_loss.item()



@hydra.main(config_path='supervised_configs/', config_name='config')
def main(cfg):
    workspace = Workspace(cfg)
    # print(cfg.pretty())
    return workspace.run()

if __name__ == '__main__':
    main()
