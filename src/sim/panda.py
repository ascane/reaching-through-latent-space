import numpy as np
# from numpy.linalg import pinv
import torch
from torch.autograd.functional import jacobian
from torch.linalg import pinv

from sim.robot3d import RoboDefinition, RoboPartDefinition, RoboCapsule
from sim.transform_matrix import z_rotation_matrix_tensor_batch


TORCH_PI = torch.acos(torch.zeros(1)).item() * 2


class Panda(RoboDefinition):
    def __init__(self):
        '''
        Panda arm DH parameters:
        https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
        https://github.com/frankaemika/franka_ros/blob/kinetic-devel/franka_description/robots/panda_arm.xacro
        '''
        parts = [
            RoboPartDefinition.create_from_dh_params(0, 0.333, 0, 0, -166, 166,
                [RoboCapsule(0.06, [0, 0, -0.333], [0, 0, -0.05])]),
            RoboPartDefinition.create_from_dh_params(0, 0, -90, 0, -101, 101,
                [RoboCapsule(0.06, [0, 0, -0.06], [0, 0, 0.06])]),
            RoboPartDefinition.create_from_dh_params(0, 0.316, 90, 0, -166, 166,
                [RoboCapsule(0.06, [0, 0, -0.22], [0, 0, -0.07])]),
            RoboPartDefinition.create_from_dh_params(0.0825, 0, 90, 0, -176, -4,
                [RoboCapsule(0.06, [0, 0, -0.06], [0, 0, 0.06])]),
            RoboPartDefinition.create_from_dh_params(-0.0825, 0.384, -90, 0, -166, 166,
                [RoboCapsule(0.06, [0, 0, -0.31], [0, 0, -0.21]),
                 RoboCapsule(0.025, [0, 0.08, -0.06], [0, 0.08, -0.20])]),
            RoboPartDefinition.create_from_dh_params(0, 0, 90, 0, -1, 215,
                [RoboCapsule(0.05, [0, 0, 0.01], [0, 0, -0.07])]),
            RoboPartDefinition.create_from_dh_params(0.088, 0, 90, 0, -166, 166,
                [RoboCapsule(0.04, [0, 0, 0.08], [0, 0, -0.06]),
                 RoboCapsule(0.03, [0.0424, 0.0424, 0.087], [0.0424, 0.0424, 0.077])]),
        ]

        root = RoboPartDefinition.chain_parts(parts)
        base_colliders = [RoboCapsule(0.06, [-0.06, 0, 0.06], [-0.09, 0, 0.06])]
        super(Panda, self).__init__(root=root, base_colliders=base_colliders, end_effector_offset=[0, 0, 0.107 + 0.0584])

        self.end_effector_offset_tensor = torch.Tensor(np.append(np.array(self.end_effector_offset).reshape(3, 1), [[1]], axis=0))
        transform_matrices = []
        current_robo_part_definition = self.root
        while current_robo_part_definition is not None:
            transform_matrices.append(current_robo_part_definition.initial_transform_matrix_tensor.unsqueeze(0))
            current_robo_part_definition = current_robo_part_definition.child
        self.transform_matrices_tensor = torch.cat(transform_matrices)

        joint_mins, joint_maxs = self.get_joint_limits()
        self.joint_min_limits_tensor = torch.tensor(joint_mins, dtype=torch.float)
        self.joint_max_limits_tensor = torch.tensor(joint_maxs, dtype=torch.float)

    def to(self, device):
        self.end_effector_offset_tensor = self.end_effector_offset_tensor.to(device)
        self.transform_matrices_tensor = self.transform_matrices_tensor.to(device)
        self.joint_min_limits_tensor = self.joint_min_limits_tensor.to(device)
        self.joint_max_limits_tensor = self.joint_max_limits_tensor.to(device)

    def FK(self, joint_angles_tensor, device, rad=False, joint_limit=True):  # in degrees
        dof = self.transform_matrices_tensor.size()[0]
        if rad:
            # Warning: This modifies joint_angles_tensor in place.
            joint_angles_tensor *= 180.0 / TORCH_PI
        joint_angles_tensor = joint_angles_tensor.view(-1, dof)
        if joint_limit:
            joint_angles_tensor = torch.max(joint_angles_tensor, self.joint_min_limits_tensor)
            joint_angles_tensor = torch.min(joint_angles_tensor, self.joint_max_limits_tensor)
        batch_size = joint_angles_tensor.size()[0]
        V = self.end_effector_offset_tensor
        V = V.repeat(batch_size, 1, 1)
        for i in range(dof - 1, -1, -1):
            V = torch.matmul(z_rotation_matrix_tensor_batch(joint_angles_tensor[:, i]).to(device), V)
            V = torch.matmul(self.transform_matrices_tensor[i].repeat(batch_size, 1, 1), V)
        return torch.squeeze(V[:,:3], dim=2)

    def jacob(self, joint_angles_tensor, device, rad=False, joint_limit=True):  # in degrees
        # joint_angles_tensor in (batch_size, dof), but only the first one is taken into account
        def fk(inputs):
            return self.FK(inputs, device, rad, joint_limit)[0]
        j = jacobian(fk, joint_angles_tensor[0])  # shape: (3, dof)
        print(j)
        return j

    def jacob_pinv(self, joint_angles_tensor, device, rad=False, joint_limit=True):  # in degrees
        jac = self.jacob(joint_angles_tensor, device, rad, joint_limit)
        return pinv(jac)
