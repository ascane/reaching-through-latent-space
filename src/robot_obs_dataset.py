from __future__ import print_function, division
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from robot_state_dataset import RobotStateDataset


EPSILON = 1e-8

TRAIN = 0
VAL = 1
TEST = 2


class RobotObstacleDataset(Dataset):

    def __init__(self, data_dir, train=TRAIN, robo_name='panda', \
                 train_data_name='collision_100k_train.dat', test_data_name='collision_10k_test.dat', \
                 free_space_train_name='free_space_100k_train.dat', free_space_test_name='free_space_10k_test.dat'):

        self.train = train
        self.robo_name = robo_name
        if robo_name == 'panda':
            self.dof = 7
        elif robo_name == 'snake':
            self.dof = 13
        self.x_dim = self.dof + 3
        self.obs_dim = 4
        self.obs_label_dim = 1

        with open(os.path.join(data_dir, train_data_name), 'rb') as f:
            np_train_val = np.load(f)

        with open(os.path.join(data_dir, test_data_name), 'rb') as f:
            np_test = np.load(f)

        size = np_train_val.shape[0]

        size_train = int(0.8 * size)
        size_val = size - size_train     
        size_test = np_test.shape[0]

        np_train = np_train_val[:size_train, :]
        np_val = np_train_val[size_train:, :]

        # normalise data
        input_dim = np_train.shape[1]
        mean_train = np_train.mean(axis=0).reshape((1, input_dim))
        std_train = np_train.std(axis=0).reshape((1, input_dim))

        # make sure robot_obs_dataset uses robot_state_dataset's mean and std.
        if free_space_train_name is not None and free_space_test_name is not None:
            val_dataset = RobotStateDataset(data_dir, train=1, train_data_name=free_space_train_name, test_data_name=free_space_test_name)
            fs_mean_train = val_dataset.get_mean_train()
            fs_std_train = val_dataset.get_std_train()

            mean_train[0:, :self.x_dim] = fs_mean_train[0, :self.x_dim]
            std_train[0:, :self.x_dim] = fs_std_train[0, :self.x_dim]

        # do not normalise if binary data
        mean_train[:, -self.obs_label_dim] = 0
        std_train[:, -self.obs_label_dim] = 1

        # If all the values are the same, set them to 0 after normalisation.
        std_train[std_train < EPSILON] = 1.0

        np_train -= np.tile(mean_train, (size_train, 1))
        np_train /= np.tile(std_train, (size_train, 1))

        np_val -= np.tile(mean_train, (size_val, 1))
        np_val /= np.tile(std_train, (size_val, 1))

        np_test -= np.tile(mean_train, (size_test, 1))
        np_test /= np.tile(std_train, (size_test, 1))
        
        if self.train == TRAIN:
            self.robot_data = torch.tensor(np_train, dtype=torch.float32)
        elif self.train == VAL:
            self.robot_data = torch.tensor(np_val, dtype=torch.float32)
        else:
            self.robot_data = torch.tensor(np_test, dtype=torch.float32)

        self.np_train = np_train
        self.np_val = np_val
        self.np_test = np_test
        self.mean_train = mean_train
        self.std_train = std_train

    def __len__(self):
        return self.robot_data.shape[0]

    def __getitem__(self, index):
        # [j1, j2, ..., jdof, ee_x, ee_y, ee_z, obs_x, obs_y, obs_h, obs_r, obs_label].
        # if collision, obs_label = 1; else, obs_label = 0.
        jpos_ee_xyz = self.robot_data[index, :self.dof + 3]
        obs_xyhr = self.robot_data[index, self.dof + 3:self.dof + 7]
        obs_label =self.robot_data[index, self.dof + 7]
        return jpos_ee_xyz, obs_xyhr, obs_label

    def get_np_train(self):
        return self.np_train

    def get_np_val(self):
        return self.np_val

    def get_np_test(self):
        return self.np_test

    def get_mean_train(self):
        return self.mean_train

    def get_std_train(self):
        return self.std_train


if __name__ == "__main__":

    data_dir = os.environ.get('RTLS_DATA')
    train_data_name='collision_10k_train.dat'
    train_data_set = RobotObstacleDataset(data_dir, train_data_name=train_data_name)

    print(train_data_set[0])
    print(len(train_data_set))

    print(train_data_set.get_mean_train())
    print(train_data_set.get_std_train())

    with open(os.path.join(data_dir, train_data_name), 'rb') as f:
        np_train = np.load(f)
        print(np_train[0])

    val_datasat_set = RobotObstacleDataset(data_dir, train=VAL, train_data_name=train_data_name)
    print(len(val_datasat_set))
