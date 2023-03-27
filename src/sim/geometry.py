import numpy as np


EPSILON = 1e-5

class Sphere(object):
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius


class Segment(object):
    def __init__(self, point1, point2):
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)


class Capsule(object):
    '''
    A capsule shape.
    radius: float - Radius of the capsule.
    point1: 3d vector - Center of the bottom sphere.
    point2: 3d vector - Center of the top sphere.
    '''
    def __init__(self, radius, point1, point2):
        self.radius = radius
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)


def dist3d_segment_to_segment(s1, s2):
    '''
    Ref: http://geomalgorithms.com/a07-_distance.html
    s1: Segment
    s2: Segment
    '''
    u = s1.point2 - s1.point1
    v = s2.point2 - s2.point1
    w = s1.point1 - s2.point1
    a = np.dot(u, u)  # always >= 0
    b = np.dot(u, v)
    c = np.dot(v, v)  # always >= 0
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a * c  - b * b  # always >= 0
    sc, sN, sD = 0, 0, D  # sc = sN / sD, default sD = D >= 0
    tc, tN, tD = 0, 0, D  # tc = tN / tD, default tD = D >= 0

    # compute the line parameters of the two closest points
    if D < EPSILON:  # the lines are almost parallel
        sN = 0.0  # force using point P0 on segment S1
        sD = 1.0  # to prevent possible division by 0.0 later
        tN = e
        tD = c
    else:  # get the closest points on the infinite lines
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:  # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:  # sc > 1  => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:  # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:  # tc > 1  => the t=1 edge is visible
        tN = tD
        # recompute sc for this edge
        if -d + b < 0.0:
            sN = 0
        elif -d + b > a:
            sN = sD
        else:
            sN = -d + b
            sD = a
    
    # finally do the division to get sc and tc
    sc = 0.0 if abs(sN) < EPSILON else sN / sD
    tc = 0.0 if abs(tN) < EPSILON else tN / tD

    # get the difference of the two closest points
    dP = w + sc * u - tc * v  # =  s1(sc) - s2(tc)

    return np.linalg.norm(dP)  # return the closest distance

def dist3d_capsule_to_capsule(c1, c2):
    s1 = Segment(c1.point1, c1.point2)
    s2 = Segment(c2.point1, c2.point2)
    return max(0, dist3d_segment_to_segment(s1, s2) - c1.radius - c2.radius)


if __name__ == "__main__":
    s1 = Segment(point1=[0,0,1], point2=[1,0,1])
    s2 = Segment(point1=[0,0,0], point2=[1,0,0])
    print(dist3d_segment_to_segment(s1, s2))
