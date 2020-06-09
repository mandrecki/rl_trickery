import gym
from .maze import MazelabEnv

gym.envs.register(
    id="Mazelab-v0", entry_point=MazelabEnv)

env_name = "Mazelab{}-v{}"
for size in [8, 12, 16, 32]:
    gym.envs.register(
        id=env_name.format(size, 0), entry_point=MazelabEnv, max_episode_steps=200,
        kwargs={
            "size":size
        })

    gym.envs.register(
        id=env_name.format(size, 1), entry_point=MazelabEnv, max_episode_steps=200,
        kwargs={
            "kind": "empty",
            "fixed": True,
            "size":size
        })

    gym.envs.register(
        id=env_name.format(size, 2), entry_point=MazelabEnv, max_episode_steps=200,
        kwargs={
            "kind": "empty",
            "fixed": False,
            "size": size
        })

    gym.envs.register(
        id=env_name.format(size, 3), entry_point=MazelabEnv,
        kwargs={
            "kind": "random",
            "fixed": True,
            "size": size
        })

    gym.envs.register(
        id=env_name.format(size, 4), entry_point=MazelabEnv,
        kwargs={
            "kind": "random",
            "fixed": False,
            "variable_goal": True,
            "size": size
        })

    gym.envs.register(
        id=env_name.format(size, 5), entry_point=MazelabEnv,
        kwargs={
            "kind": "random",
            "fixed": False,
            "size": size
        })
