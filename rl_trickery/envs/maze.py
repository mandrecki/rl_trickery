import numpy as np
import pkg_resources

from gym.spaces import Box
from gym.spaces import Discrete

from mazelab import BaseEnv
from mazelab import VonNeumannMotion
from mazelab import BaseMaze
from mazelab import Object
from mazelab.generators.random_maze import random_maze
from mazelab.solvers.dijkstra_solver import dijkstra_solver, dijkstra_solver_full
# from mazelab import DeepMindColor as BoardColor


class BoardColor:
    obstacle = (0, 0, 0)
    free = (0, 100, 0)
    agent = (0, 0, 255)
    goal = (255, 0, 0)


def generate(kind, size=12):
    if kind == "empty":
        return generate_empty(size)
    elif kind == "maze":
        return generate_random(size)
    else:
        raise ValueError("Bad maze kind {}".format(kind))


def generate_random(size=12):
    x = random_maze(width=size, height=size, complexity=.1, density=.3)
    return x


def generate_empty(size=12):
    x = np.zeros((size, size))
    x[:, 0] = 1
    x[:, -1] = 1
    x[0, :] = 1
    x[-1, :] = 1
    return x


class Maze(BaseMaze):
    def __init__(self, kind="empty", size=12):
        self.start = np.array([[0,0]])
        self.goal = np.array([[0,0]])
        self.board = generate(kind, size)
        self.randomize_agent()
        self.randomize_goal()
        super().__init__()

    @property
    def size(self):
        return self.board.shape

    def make_objects(self):
        free = Object('free', 0, BoardColor.free, False, np.stack(np.where(self.board == 0), axis=1))
        obstacle = Object('obstacle', 1, BoardColor.obstacle, True, np.stack(np.where(self.board == 1), axis=1))
        agent = Object('agent', 2, BoardColor.agent, False, [])
        goal = Object('goal', 3, BoardColor.goal, False, [])
        return free, obstacle, agent, goal

    def randomize_agent(self):
        legal = np.array(np.where(self.board == 0))
        agent_idx = self.goal
        while (self.goal == agent_idx).all():
            agent_idx = np.atleast_2d(legal[:, np.random.randint(0, legal.shape[1])])
        self.start = agent_idx
        return self.start

    def randomize_goal(self):
        legal = np.array(np.where(self.board == 0))
        goal_idx = self.start
        while (self.start == goal_idx).all():
            goal_idx = np.atleast_2d(legal[:, np.random.randint(0, legal.shape[1])])
        self.goal = goal_idx
        return self.goal

    def default_goal(self):
        self.goal = np.atleast_2d(np.unravel_index(np.argmin(self.board, axis=None), self.board.shape))

    def from_numpy(self, array):
        self.board = array
        self.randomize_agent()
        self.randomize_goal()
        super().__init__()
        return self


class MazelabEnv(BaseEnv):
    def __init__(self, maze_size, maze_kind, goal_fixed, maze_fixed, goal_reward, wall_reward, **kwargs):
        super().__init__()
        self.size = maze_size
        self.kind = maze_kind
        self.maze_fixed = maze_fixed
        self.goal_fixed = goal_fixed
        self.goal_reward = goal_reward
        self.wall_reward = wall_reward

        if self.maze_fixed and self.kind == "maze":
            file = pkg_resources.resource_filename("rl_trickery", "envs/mazes/raw_maze_{}.npy".format(int(self.size)))
            array = np.load(file)
            self.maze = Maze("empty", 8).from_numpy(array)
        else:
            self.maze = Maze(self.kind, self.size)

        if self.goal_fixed:
            self.maze.default_goal()

        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=255, shape=list(self.maze.size)+[3], dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        reward = 0.0
        if valid:
            self.maze.objects.agent.positions = [new_position]
        else:
            reward += -0.01 * int(self.wall_reward)

        if self._is_goal(new_position):
            reward += 1.0 * int(self.goal_reward)
            done = True
        else:
            reward += -0.01
            done = False
        return self.maze.to_rgb(), reward, done, {}

    def seed(self, seed=None):
        s = np.random.seed(seed)
        return s

    def reset(self):
        if not self.maze_fixed and self.kind != "empty":
            self.maze = Maze(self.kind, self.size)

        if not self.goal_fixed:
            self.maze.randomize_goal()

        self.maze.objects.goal.positions = self.maze.goal
        self.maze.objects.agent.positions = self.maze.randomize_agent()
        return self.maze.to_rgb()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()

    def get_distance_to_goal(self):
        current_position = self.maze.objects.agent.positions[0]
        actions = dijkstra_solver(self.maze.board.astype("bool"), self.motions, current_position.flatten(), self.maze.goal.flatten())
        distance = len(actions)
        return distance

    def get_distance_matrix(self):
        current_position = self.maze.objects.agent.positions[0]
        distances = dijkstra_solver_full(self.maze.board.astype("bool"), self.motions, current_position.flatten(), self.maze.goal.flatten())
        distances[distances > 1000] = -1
        return distances
