import numpy as np
from scipy.spatial.transform import Rotation as R

from sim.geometry import Capsule, dist3d_capsule_to_capsule
from sim.object3d import Object3D
from sim.transform_matrix import transform_matrix_tensor


FMAX = np.finfo('float').max


class Scene(object):
    '''
    robo: Robo3D
    target: Sphere
    obstacles: [Sphere]
    '''
    def __init__(self, robo, target, obstacles):
        self.robo = robo
        self.target = target
        self.obstacles = obstacles


class Robo3D(object):
    '''
    root: RoboPart
    base_colliders: [RoboCapsule]
    end_effector_offset: 3d vector
    state: RoboState
    '''
    def __init__(self, definition, state=None):
        '''
        definition: RoboDefinition
        '''
        self.root = RoboPart(definition.root)
        self.base_colliders = definition.base_colliders
        self.end_effector_offset = definition.end_effector_offset
        if state is None:
            state = RoboState(hinge_angles=[0] * definition.get_dof())
        self.update_state(state)

    def update_state(self, state):
        self.state = state
        current_robo_part = self.root
        for theta in state.hinge_angles:
            current_robo_part.update_angle(theta)
            current_robo_part = current_robo_part.child

    def get_state(self):
        return self.state

    def get_ee_xyz(self):
        return self.root.get_deepest_child().transform_point(self.end_effector_offset)

    def get_colliders(self):
        '''
        colliders: [Capsule]
        '''
        colliders = []
        # add base colliders
        for c in self.base_colliders:
            colliders.append(Capsule(c.radius, c.point1, c.point2))
        # add other colliders
        current_robo_part = self.root
        while current_robo_part is not None:
            for robo_capsule in current_robo_part.definition.colliders:
                point1 = current_robo_part.transform_point(robo_capsule.point1)
                point2 = current_robo_part.transform_point(robo_capsule.point2)
                colliders.append(Capsule(robo_capsule.radius, point1, point2))
            current_robo_part = current_robo_part.child
        return colliders

    def dist_to_obstacles(self, obstacles):
        '''
        obstacles: [Capsule]
        '''
        min_dist = FMAX
        colliders = self.get_colliders()
        for collider in colliders:
            for obstacle in obstacles:
                min_dist = min(min_dist, dist3d_capsule_to_capsule(collider, obstacle))
        return min_dist

    def dist_jpos_to_obstacles(self, jpos, obstacles_xyhr):
        '''
        jpos: [j1, j2, j3, ..., j7]
        obs_config: [[x1, y1, h1, r1], [x2, y2, h2, r2], ...]
        '''
        state = RoboState(jpos)
        self.update_state(state)
        obstacles = []
        for obs_config in obstacles_xyhr:
            x, y, h, r = obs_config
            obstacles.append(Capsule(radius=r, point1=[x, y, 0], point2=[x, y, h]))
        return self.dist_to_obstacles(obstacles)

    def check_for_collision(self, jpos, obstacles_xyhr):
        '''
        jpos: [j1, j2, j3, ..., j7]
        obs_config: [[x1, y1, h1, r1], [x2, y2, h2, r2], ...]
        '''
        return self.dist_jpos_to_obstacles(jpos, obstacles_xyhr) == 0


class RoboPart(Object3D):
    '''
    definition: RoboPartDefinition
    parent: RoboPart
    '''
    def __init__(self, definition, parent=None):
        super(RoboPart, self).__init__(parent=parent)
        self.definition = definition
        self.child = None

        self._construct_recursively()

    def _construct_recursively(self):
        self.local_rotation = self.definition.initial_local_rotation
        self.local_position = self.definition.initial_local_position

        if self.definition.child is not None:
            self.child = RoboPart(definition=self.definition.child, parent=self)

    def get_deepest_child(self):
        if self.child is None:
            return self
        else:
            return self.child.get_deepest_child()

    def get_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

    def update_angle(self, theta):
        '''
        theta: in degrees
        '''
        if theta < self.definition.min_angle:
            theta = self.definition.min_angle
        if theta > self.definition.max_angle:
            theta = self.definition.max_angle
        theta = theta / 180.0 * np.pi
        axis = self.definition.hinge_axis_local_space
        quat = np.append(np.sin(theta/2) * axis, [np.cos(theta/2)])
        self.local_rotation = self.definition.initial_local_rotation * R.from_quat(quat)


class RoboDefinition(object):
    '''
    The definition of the robot.
    root: RoboPartDefinition - Definition of the root part.
    base_colliders: [RoboCapsule] - A listing of capsules approximating the 3D shape of the non-movable base of the robot.
                                    Coordinates are in local space.
    end_effector_offset: 3d vector - Coordinates of the end effector in the reference frame of the deepest child part.
    '''
    def __init__(self, root, base_colliders, end_effector_offset):
        self.root = root
        self.base_colliders = base_colliders
        self.end_effector_offset = end_effector_offset

    def get_dof(self):
        '''
        Gets degree of freedom (i.e. number of links).
        '''
        current_robo_part_definition = self.root
        count = 0
        while current_robo_part_definition is not None:
            count += 1
            current_robo_part_definition = current_robo_part_definition.child
        return count

    def get_joint_limits(self):
        mins = []
        maxs = []
        current_robo_part_definition = self.root
        while current_robo_part_definition is not None:
            mins.append(current_robo_part_definition.min_angle)
            maxs.append(current_robo_part_definition.max_angle)
            current_robo_part_definition = current_robo_part_definition.child
        return mins, maxs


class RoboPartDefinition(object):
    '''
    child: RoboPartDefinition - Definition of the next part.
    initial_local_position: 3d vector - The initial local position of the part. (i.e. coordinates of the local origin in parent space).
    initial_local_rotation: 3d vector - The initial local rotation of the part. (local rotation * coords local space = coords parent space).
    hinge_axis_local_space: 3d vector - The hinge rotation axis direction in local space.
    min_angle: float - minimum joint value in degrees.
    max_angle: float - maximum joint value in degrees.
    colliders: [RoboCapsule] - A listing of capsules approximating the 3D shape of this arm part. Coordinates are in local space.
    '''
    def __init__(self, initial_local_position, initial_local_rotation, hinge_axis_local_space, min_angle=-180.0, max_angle=180.0, colliders=[], child=None):
        self.child = child
        self.initial_local_position = initial_local_position
        self.initial_local_rotation = initial_local_rotation
        self.hinge_axis_local_space = hinge_axis_local_space / np.linalg.norm(hinge_axis_local_space)  # normalise
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.colliders = colliders
        self.initial_transform_matrix_tensor = transform_matrix_tensor(initial_local_rotation, initial_local_position)

    @staticmethod
    def create_from_dh_params(r, d, alpha, theta, min_theta=-180.0, max_theta=180.0, colliders=[]):
        '''
        Creates the part definition from Modified DH parameters. Only initial_local_position, initial_local_rotation,
        and hinge_axis_local_space (always (0,0,1)) are populated.
        (see https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters)
        r: float - r value (translation along x axis) of the link connecting the previous part to this part [r_(i-1)i].
        d: float - d value (translation along z axis) of the current hinge axis at the origin of this part [d_i].
        alpha: float - alpha value (rotation around x axis) of the link connecting the previous part to this part [alpha_(i-1)i].
        theta: float - theta value (rotation around z axis) of the current hinge axis at the origin of this part [theta_i].
        '''
        return RoboPartDefinition(
            initial_local_position=R.from_euler('x', alpha, degrees=True).apply([r, 0, 0] + R.from_euler('z', theta, degrees=True).apply([0, 0, d])),
            initial_local_rotation=R.from_euler('x', alpha, degrees=True) * R.from_euler('z', theta, degrees=True),
            hinge_axis_local_space=[0, 0, 1],
            min_angle=min_theta,
            max_angle=max_theta,
            colliders=colliders)

    @staticmethod
    def chain_parts(parts):
        '''
        parts: [RoboPartDefinition]
        '''
        parts = np.asarray(parts)
        for i in range(len(parts)-1):
            parts[i].child = parts[i+1]
        return parts[0]


class RoboCapsule(object):
    '''
    A capsule shape, used as colliders of the robot part.
    radius: float - Radius of the capsule.
    point1: 3d vector - Center of the bottom sphere.
    point2: 3d vector - Center of the top sphere.
    '''
    def __init__(self, radius, point1, point2):
        self.radius = radius
        self.point1 = point1
        self.point2 = point2


class RoboState(object):
    '''
    State of the entire robot arm.
    hinge_angles: [float] - Joint angles (theta value in DH) from the root to the deepest child.
    '''
    def __init__(self, hinge_angles):
        self.hinge_angles = hinge_angles
