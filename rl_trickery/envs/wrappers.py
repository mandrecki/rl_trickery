import numpy as np
import gym
import cv2
import torch
from baselines.common.vec_env import VecEnvWrapper, VecEnvObservationWrapper
from baselines.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv, ClipRewardEnv, NoopResetEnv, MaxAndSkipEnv


def wrap_deepmind_modified(env, episode_life=False, clip_rewards=False, to_grayscale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if to_grayscale:
        env = ToGrayscale(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env


# Standardising environments
class ToImageObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(ToImageObservation, self).__init__(env)
        self.reset()
        image = self.render(mode='rgb_array')
        image_size = image.shape[0:2]
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*image_size, channels],
                                                dtype=np.uint8)

    def observation(self, symbolic_observation):
        image = self.render(mode='rgb_array')
        return image


class CropImage(gym.ObservationWrapper):
    def __init__(self, env, crop_box):
        super(CropImage, self).__init__(env)
        self.y_low, self.y_high, self.x_low, self.x_high = crop_box
        image_size = (self.y_high - self.y_low, self.x_high - self.x_low)
        channels = 3
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*image_size, channels],
                                                dtype=np.uint8)

    def observation(self, image):
        image = image[self.y_low: self.y_high, self.x_low: self.x_high]
        return image


class ToGrayscale(gym.ObservationWrapper):
    def __init__(self, env):
        super(ToGrayscale, self).__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[0], old_shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        return frame


class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, new_size, antialias=False):
        super(ResizeImage, self).__init__(env)
        self.new_size = new_size
        self.antialias = antialias
        self.channels = env.observation_space.shape[2]
        self.observation_space = gym.spaces.Box(0, 255,
                                                [*self.new_size, self.channels],
                                                dtype=np.uint8)

    def add_padding(self, image):
        paddings = []
        for old_size in image.shape[0:2]:
            full_padding = 2 ** np.ceil(np.log2(old_size)) - old_size
            start_padding = int(np.floor(full_padding/2))
            end_padding = int(np.ceil(full_padding/2))
            paddings.append(start_padding)
            paddings.append(end_padding)

        if any(paddings):
            image = cv2.copyMakeBorder(image, *paddings, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image

    def observation(self, image):
        # only add padding if starting with small image
        if self.antialias and image.shape[0] != self.new_size[0]:
            image = self.add_padding(image)
            image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_NEAREST)
        else:
            image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_AREA)

        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

        return image


class RandomPadCropImage(gym.ObservationWrapper):
    def __init__(self, env, padding=4):
        super(RandomPadCropImage, self).__init__(env)
        self.padding = padding
        self.x0, self.y0 = padding, padding

    def add_padding(self, image):
        image = cv2.copyMakeBorder(image, *tuple(4*[self.padding]), cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image

    def reset(self):
        self.x0, self.y0 = np.random.randint(0, 2*self.padding, 2)
        return self.observation(self.env.reset())

    def observation(self, image):
        image = self.add_padding(image)
        image = image[self.y0: (self.y0-2*self.padding), self.x0: (self.x0-2*self.padding)]
        return image


class ScaleImage(gym.ObservationWrapper):
    def observation(self, image):
        return image/255.0


class StepSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super(StepSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class RandomResetSteps(gym.Wrapper):
    def __init__(self, env, max_random_steps=30):
        super(RandomResetSteps, self).__init__(env)
        self.max_random_steps = max_random_steps

    def reset(self):
        obs = self.env.reset()
        for i in range(np.random.randint(0, self.max_random_steps)):
            obs, reward, done, info = self.env.step(self.env.action_space.sample())
            if done:
                obs = self.env.reset()

        return obs


class PauseWrapper(gym.Wrapper):
    def __init__(self, env, special_value=127):
        super(PauseWrapper, self).__init__(env)
        self.obs = None
        self.rew = 0
        self.done = False
        self.info = {}
        self.special_value = special_value
        assert env.action_space.__class__.__name__ == "Discrete"

    def step(self, action):
        if action != self.special_value:
            self.obs, self.rew, self.done, self.info = self.env.step(action)
        else:
            self.rew = 0
            self.info = {}

        return self.obs, self.rew, self.done, self.info

    def reset(self):
        self.obs = self.env.reset()
        self.rew = 0
        self.done = False
        self.info = {}
        return self.obs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
