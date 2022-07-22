import numpy as np
from numpy import cos, sin


def R(theta, u):
    return [[cos(theta) + u[0]**2 * (1-cos(theta)),
             u[0] * u[1] * (1-cos(theta)) - u[2] * sin(theta),
             u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
            [u[0] * u[1] * (1-cos(theta)) + u[2] * sin(theta),
             cos(theta) + u[1]**2 * (1-cos(theta)),
             u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
            [u[0] * u[2] * (1-cos(theta)) - u[1] * sin(theta),
             u[1] * u[2] * (1-cos(theta)) + u[0] * sin(theta),
             cos(theta) + u[2]**2 * (1-cos(theta))]]


def Rotate(pointToRotate, point1, point2, theta):
    u= []
    squaredSum = 0
    for i,f in zip(point1, point2):
        u.append(f-i)
        squaredSum += (f-i) **2

    u = [i/squaredSum for i in u]

    r = R(theta, u)
    rotated = []

    for i in range(3):
        rotated.append(round(sum([r[j][i] * pointToRotate[j] for j in range(3)]), 4))

    return np.array(rotated)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))