import gym
from .maze import MazelabEnv


env_id = "Mazelab-v0"
gym.envs.register(
    id=env_id, entry_point=MazelabEnv, max_episode_steps=200,
    kwargs={
    })

env_id = "Mazelab-v1"
gym.envs.register(
    id=env_id, entry_point=MazelabEnv, max_episode_steps=200,
    kwargs={
        "kind": "empty",
        "fixed": True,
    })

env_id = "Mazelab-v2"
gym.envs.register(
    id=env_id, entry_point=MazelabEnv, max_episode_steps=500,
    kwargs={
        "kind": "empty",
        "fixed": False,
    })

env_id = "Mazelab-v3"
gym.envs.register(
    id=env_id, entry_point=MazelabEnv, max_episode_steps=500,
    kwargs={
        "kind": "random",
        "fixed": True,
    })

env_id = "Mazelab-v4"
gym.envs.register(
    id=env_id, entry_point=MazelabEnv, max_episode_steps=500,
    kwargs={
        "kind": "random",
        "fixed": False,
        "variable_goal": True,
    })

env_id = "Mazelab-v5"
gym.envs.register(
    id=env_id, entry_point=MazelabEnv, max_episode_steps=500,
    kwargs={
        "kind": "random",
        "fixed": False,
    })
