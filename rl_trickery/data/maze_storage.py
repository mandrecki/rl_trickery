import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from rl_trickery.envs.maze import Maze


def generate_dataset(env, n):
    boards = []
    distances = []
    goals = []
    for i in range(int(n)):
        env.reset()
        boards.append(torch.IntTensor(env.unwrapped.maze.board))
        distances.append(torch.IntTensor(env.get_distance_matrix().reshape(env.unwrapped.maze.board.shape)))
        goals.append(torch.IntTensor(env.unwrapped.maze.goal))

    boards = np.stack(boards)
    distances = np.stack(distances)
    goals = np.stack(goals)
    ds = MazeDataset(boards, distances, goals)
    return ds


class MazeDataset(Dataset):
    def __init__(self, boards, distances, goals):
        super(MazeDataset, self).__init__()
        self.boards = boards
        self.distances = distances
        self.goals = goals

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, index):
        board = self.boards[index]
        distances = self.distances[index]
        goal = self.goals[index]

        m = Maze().from_numpy(board)
        m.make_objects()
        m.goal = goal
        m.objects.goal.positions = goal
        m.objects.agent.positions = m.randomize_agent()
        pos_x, pos_y = m.objects.agent.positions.T

        image = m.to_rgb()
        image = self.add_padding(image)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image) / 255

        y = torch.FloatTensor(distances[pos_x, pos_y])

        return image, y

    def add_padding(self, image):
        paddings = []
        for old_size in image.shape[0:2]:
            full_padding = 2 ** np.ceil(np.log2(old_size)) - old_size
            start_padding = int(np.floor(full_padding / 2))
            end_padding = int(np.ceil(full_padding / 2))
            paddings.append(start_padding)
            paddings.append(end_padding)

        if any(paddings):
            image = cv2.copyMakeBorder(image, *paddings, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image
