import numpy as np
import math

# calculates rotation matrix around an axis
def rotmat(theta, u):
    # make sure its unit vector
    u = u / np.linalg.norm(u)

    # rotation matrix
    R = np.arr([[(1 - math.cos(theta)) * (u[0] ** 2) + math.cos(theta),
                (1 - math.cos(theta)) * u[0] * u[1] - math.sin(theta) * u[2],
                (1 - math.cos(theta)) * u[0] * u[2] + math.sin(theta) * u[1]],
                [(1 - math.cos(theta)) * u[1] * u[0] + math.sin(theta) * u[2],
                (1 - math.cos(theta)) * (u[1] ** 2) + math.cos(theta),
                (1 - math.cos(theta)) * u[1] * u[2] - math.sin(theta) * u[0]],
                [(1 - math.cos(theta)) * u[2] * u[0] - math.sin(theta) * u[1],
                (1 - math.cos(theta)) * u[2] * u[1] + math.sin(theta) * u[0],
                (1 - math.cos(theta)) * (u[2] ** 2) + math.cos(theta)]])

    return R


# rotates and moves point c_p
def RotateTranslate(c_p, theta, u, A, t):
    # get matrix
    R = rotmat(theta, u)

    # L^(-1)
    R_inv = np.transpose(R)

    # translate
    c_q = np.dot(R_inv, c_p) + t

    return c_q

# calculates new coordinates of c_p
# when we move and rotate start point of system
def ChangeCoordinateSystem(c_p, R, c_0):
    # L^(-1)
    R_inv = np.transpose(R)

    # new coordinates
    d_p = np.dot(R_inv, c_p - c_0)

    return d_p

