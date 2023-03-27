import numpy as np
import torch

def transform_matrix(rotation, translation):
    '''
    rotation: scipy.spatial.transform.rotation.Rotation
    translation: numpy array of shape (3,)
    '''
    transform_matrix = np.zeros((4,4))
    # R = rotation.as_dcm()
    R = rotation.as_matrix()
    for i in range(3):
        for j in range(3):
            transform_matrix[i, j] = R[i, j]
    for i in range(3):
        transform_matrix[i, 3] = translation[i]
    transform_matrix[3, 3] = 1

    return transform_matrix

def transform_matrix_tensor(rotation, translation):
    return torch.Tensor(transform_matrix(rotation, translation))

def z_rotation_matrix(angle):
    theta = angle / 180.0 * np.pi
    return np.array([[np.cos(theta), -np.sin(theta), 0 , 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def z_rotation_matrix_tensor(angle_scalar_tensor):
    theta = angle_scalar_tensor / 180.0 * np.pi
    t = torch.zeros([4, 4], dtype=torch.float)
    t[0,0] = torch.cos(theta)
    t[0,1] = -torch.sin(theta)
    t[1,0] = torch.sin(theta)
    t[1,1] = torch.cos(theta)
    t[2,2] = 1.
    t[3,3] = 1.
    return t

def z_rotation_matrix_tensor_batch(angle_batch_tensor):
    theta_batch = angle_batch_tensor / 180.0 * np.pi
    batch_size = theta_batch.size()[0]
    t = torch.zeros([batch_size, 4, 4], dtype=torch.float)
    t[:,0,0] = torch.cos(theta_batch)
    t[:,0,1] = -torch.sin(theta_batch)
    t[:,1,0] = torch.sin(theta_batch)
    t[:,1,1] = torch.cos(theta_batch)
    t[:,2,2] = 1.
    t[:,3,3] = 1.
    return t