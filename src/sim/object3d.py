from scipy.spatial.transform import Rotation as R


class Object3D(object):
    '''
    parent: Object3D
    local_position: 3d vector
    local_rotation: Rotation
    '''
    def __init__(self, parent=None, local_position=[0,0,0], local_rotation=R.from_quat([0,0,0,1])):
        self.parent = parent
        self.local_position = local_position
        self.local_rotation = local_rotation

    def get_world_position(self):
        world_position = self.transform_point([0, 0, 0])
        return world_position

    def get_world_rotation(self):
        if self.parent is None:
            return self.local_rotation
        else:
            return self.parent.get_world_rotation() * self.local_rotation

    def transform_point(self, position_in_local_space):
        '''
        position_in_local_space: 3d vector
        '''
        position_in_parent_space = self.local_rotation.apply(position_in_local_space) + self.local_position
        if self.parent is None:
            return position_in_parent_space
        else:
            return self.parent.transform_point(position_in_parent_space)

    def transform_vector(self, vector_in_local_space):
        '''
        vector_in_local_space: 3d vector
        '''
        vector_in_parent_space = self.local_rotation.apply(vector_in_local_space)
        if self.parent is None:
            return vector_in_parent_space
        else:
            return self.parent.transform_vector(vector_in_parent_space)

    def inverse_transfrom_point(self, position_in_world_space):
        '''
        position_in_world_space: 3d vector
        '''
        if self.parent is None:
            position_in_parent_space = position_in_world_space
        else:
            position_in_parent_space = \
                self.parent.inverse_transfrom_point(position_in_world_space)
        return self.local_rotation.inv().apply(position_in_parent_space - self.local_position)

    def inverse_transform_vector(self, vector_in_world_space):
        '''
        vector_in_world_space: 3d vector
        '''
        if self.parent is None:
            vector_in_parent_space = vector_in_world_space
        else:
            vector_in_parent_space = \
                self.parent.inverse_transform_vector(vector_in_world_space)
        return self.local_rotation.inv().apply(vector_in_parent_space)
