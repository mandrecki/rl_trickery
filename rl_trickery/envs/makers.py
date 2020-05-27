import gym
import numpy as np

from baselines.common.vec_env import DummyVecEnv, VecEnvWrapper, SubprocVecEnv
from baselines import bench


from .wrappers import ToImageObservation, CropImage, ResizeImage, RandomPadCropImage, ScaleImage, TransposeImage, VecPyTorch, \
    VecPyTorchFrameStack

env_id = "EmptyMazelab-v0"


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

UPSCALE_ENVS = [
    "EmptyMazelab-v0",
]

CROP_ENVS = {
}

IM_SIZE = 64


def make_env(
        env_id,
        *extra_args,
        seed=0,
        max_episode_length=200,
        pytorch_dim_order=True,
        target_size=(IM_SIZE, IM_SIZE),
        augment=False
):
    if "Mazelab" in env_id:
        env = gym.make(env_id)
    elif env_id in GYM_ENVS:
        env = gym.make(env_id)
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
            height=IM_SIZE,
            width=IM_SIZE,
            frame_skip=CONTROL_SUITE_ACTION_REPEATS[domain_name],
            camera_id=camera_id,
            channels_first=False
        )

    else:
        raise ValueError("Bad env_id", env_id)

    # Crop and resize if necessary
    if env_id in CROP_ENVS.keys():
        env = CropImage(env, CROP_ENVS.get(env_id))
    if env_id in UPSCALE_ENVS:
        env = ResizeImage(env, target_size, antialias=True)
    elif env.observation_space.shape[0:2] != target_size:
        env = ResizeImage(env, target_size, antialias=False)

    if augment:
        env = RandomPadCropImage(env)
    env = ScaleImage(env)

    if pytorch_dim_order:
        env = TransposeImage(env)

    env = bench.Monitor(env, filename=None, allow_early_resets=True)

    return env


def env_generator(env_id, seed=0, target_size=(IM_SIZE, IM_SIZE), **kwargs):
    def _thunk():
        env = make_env(env_id, seed=seed, pytorch_dim_order=True, target_size=target_size, **kwargs)
        return env

    return _thunk


def make_envs(env_id, seed, n_envs, device, frame_stack=1):
    envs = [env_generator(env_id, seed=seed + 1000 * i) for i in range(n_envs)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if frame_stack > 1:
        envs = VecPyTorchFrameStack(envs, frame_stack, device)

    return envs