import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import gym
import gym_tetris
import gym_ple
import gym_minigrid
import gym_minipacman
import gym_minatar
import gym_pygame
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import TimeLimit

import numpy as np

from baselines.common.vec_env import DummyVecEnv, VecEnvWrapper, SubprocVecEnv
from baselines import bench

from baselines.common.atari_wrappers import make_atari, wrap_deepmind


from .wrappers import *



ABSTRACT_ENVS = [
    "EmptyMazelab-v0",
]

GYM_ENVS = [ "CartPole-v0",
    "Pendulum-v0", "MountainCar-v0", "Ant-v2", "HalfCheetah-v2", "HalfCheetah-v2", "Humanoid-v2",
    "HumanoidStandup-v2", "InvertedDoublePendulum-v2", "InvertedPendulum-v2", "Reacher-v2", "Swimmer-v2",
    "Walker2d-v2"
]

DMC2_ENVS = [
    "cartpole-balance", "cartpole-swingup", "reacher-easy", "finger-spin", "cheetah-run",
    "ball_in_cup-catch", "walker-walk"
]
CONTROL_SUITE_ACTION_REPEATS = {"cartpole": 8, "reacher": 4, "finger": 2, "cheetah": 4, "ball_in_cup": 2, "walker": 2}

MINIPACMAN_ENVS = [
    "RegularMiniPacmanNoFrameskip-v0",
    "AvoidMiniPacmanNoFrameskip-v0",
    "HunMiniPacmanNoFrameskip-v0",
    "AmbushMiniPacmanNoFrameskip-v0",
    "RushMiniPacmanNoFrameskip-v0",
]

UPSCALE_ENVS = [
    "Mazelab-v0",
] # + MINIPACMAN_ENVS

CROP_ENVS = {
    "TetrisA-v2": (45, 211, 93, 178),  # only blocks visible
}


def make_env(
        env_id,
        env_kwargs={},
        seed=0,
        pytorch_dim_order=True,
        obs_type="image",
        image_size=(84, 84),
        frame_skip=1,
        cognitive_pause=False,
        random_initial_steps=0,
        max_timesteps=None,
        episode_life=False,
        to_grayscale=False,
        clip_rewards=False,
        **kwargs
):
    env = gym.make(env_id, **env_kwargs)
    if env_id in GYM_ENVS and obs_type == "image":
        env = ToImageObservation(env)
    elif env_id in DMC2_ENVS:
        import dmc2gym
        domain_name, task_name = env_id.split("-")
        camera_id = 2 if domain_name == 'quadruped' else 0
        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            visualize_reward=False,
            from_pixels=True,
            height=image_size,
            width=image_size,
            frame_skip=CONTROL_SUITE_ACTION_REPEATS[domain_name],
            camera_id=camera_id,
            channels_first=False
        )
    elif env_id == "TetrisA-v2":
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env.seed(seed)
    if random_initial_steps > 0:
        env = RandomResetSteps(env, random_initial_steps)
    if frame_skip > 1:
        env = StepSkipEnv(env, skip=frame_skip)
    if max_timesteps:
        env = TimeLimit(env, max_episode_steps=max_timesteps)
    env = bench.Monitor(env, filename=None, allow_early_resets=True)

    if episode_life:
        env = EpisodicLifeEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)

    image_size = tuple(image_size)
    # if image env
    if obs_type == "image":
        assert len(env.observation_space.shape) == 3
        if to_grayscale:
            env = ToGrayscale(env)
        # Crop and resize if necessary
        if env_id in CROP_ENVS.keys():
            env = CropImage(env, CROP_ENVS.get(env_id))
        if env_id in UPSCALE_ENVS:
            env = ResizeImage(env, image_size, antialias=True)
        elif env.observation_space.shape[0:2] != image_size:
            env = ResizeImage(env, image_size, antialias=False)
        if pytorch_dim_order:
            env = TransposeImage(env)
    elif obs_type == "proprioceptive":
        pass
    else:
        raise NotImplementedError

    if cognitive_pause:
        env = PauseWrapper(env, cognitive_pause)

    return env


def env_generator(env_id, seed=0, **kwargs):
    def _thunk():
        env = make_env(env_id, seed=seed, **kwargs)
        return env

    return _thunk


def make_envs(env_id, device, seed=0, num_envs=1, frame_stack=1, **kwargs):
    envs = [env_generator(env_id, seed=seed + 1000 * i, **kwargs) for i in range(num_envs)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if frame_stack > 1:
        envs = VecPyTorchFrameStack(envs, frame_stack, device)

    return envs