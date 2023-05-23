import numpy as np
import math
from edge import Edge

# calculates rotation matrix around an axis
def rotmat(theta, u):
    # make sure its unit vector
    u = u / np.linalg.norm(u)

    # rotation matrix
    R = np.array([[(1 - math.cos(theta)) * (u[0] ** 2) + math.cos(theta),
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

    # move to A
    c_p = c_p.T - A

    # rotate
    c_p = np.dot(R, c_p.T)

    # move back to starting system
    c_q = c_p.T + A

    # displace by t
    c_q = c_q + t

    return c_q.T


# calculates new coordinates of c_p
# when we move and rotate start point of system
def ChangeCoordinateSystem(c_p, R, c_0):
    # L^(-1)
    R_inv = R.T

    # new coordinates
    # transposes are primarily used cause of python
    c_temp = c_p.T - c_0
    d_p = np.dot(R_inv, c_temp.T)

    return d_p


# projects point in wcs to ccs using pinhole model
def PinHole(f, cv, cx, cy, cz, p3d):
    # rotation matrix
    R = np.array([cx, cy, cz]).T

    # change coordinate system to ccs
    p3d_ccs = ChangeCoordinateSystem(p3d, R, cv)

    # extract z coordinate
    depth = p3d_ccs[2, :].T.flatten()

    # project
    x_proj = f * p3d_ccs[0, :] / depth
    y_proj = f * p3d_ccs[1, :] / depth
    p2d = np.array([x_proj[0, :], y_proj[0,:]])

    return p2d, depth


# projects the same without knowing ccs unit vectors
def CameraLookingAt(f, cv, cK, cup, p3d):
    # calculate unit vectors
    cz = cK / np.linalg.norm(cK)

    t = cup - np.dot(cup.T,  cz) * cz
    cy = t / np.linalg.norm(t)

    cy = cy.T
    cz = cz.T

    cx = np.cross(cy, cz)

    # use pinhole
    p2d, depth = PinHole(f, cv.T, cx, cy, cz, p3d)

    return p2d, depth


# maps coords of projected points to pixels
def rasterize(p2d, Rows, Columns, H, W):
    # scale to new range
    # + (H/w)/2 because camera coords are in range [-H/2, H/2]
    n2d = np.array([(p2d[0, :] + H / 2) * (Rows - 1) / H, \
                    (p2d[1, :] + W / 2) * (Columns - 1) / W])

    # round
    n2d = np.round(n2d).astype(int)

    return n2d


# interpolation for colors
def interpolate_vectors(p1, p2, V1, V2, xy, dim):
    if dim == 1:
        if p1[0] == p2[0]:
            return V1
        l = (xy - p1[0]) / (p2[0] - p1[0])
    else:
        if p1[1] == p2[1]:
            return V1
        l = (xy - p1[1]) / (p2[1] - p1[1])

    V = (1 - l) * V1 + l * V2
    V = np.clip(V, 0, 1)

    return V


# implements gouraud shading
def Gourauds(canvas, vertices, vcolors):
    updatedcanvas = canvas

    if all(vertices[0] == vertices[1]) and all(vertices[1] == vertices[2]):
        updatedcanvas[vertices[0, 1], vertices[0, 0]] = np.mean(vcolors, axis=0)
        return updatedcanvas

    edges = [Edge() for _ in range(3)]
    edges[0] = Edge(np.array([vertices[0], \
                                 vertices[1]]))
    edges[1] = Edge(np.array([vertices[1], \
                                 vertices[2]]))
    edges[2] = Edge(np.array([vertices[2], \
                                 vertices[0]]))

    y_min = min([edge.y_min[1] for edge in edges])
    y_max = max([edge.y_max for edge in edges])

    actives = 0
    for edge in edges:
        if edge.y_min[1] == y_min:
            edge.active = True
            actives = actives + 1
    border_points = []

    if actives == 3:
        for edge in edges:
            if edge.m == float('-inf'):
                edge.active = False
                actives = actives - 1

    if actives == 3:
        for i, edge in enumerate(edges):
            if edge.m == 0:
                for x in range(edge.vertices[0][0], edge.vertices[1][0] + 1):
                    updatedcanvas[y_min, x] = interpolate_vectors(edge.vertices[0], edge.vertices[1],\
                                                           vcolors[i], vcolors[(i + 1) % 3], \
                                                           x, 1)
                actives = actives - 1
                edge.active = False
            else:
                border_points.append([edge.y_min[0] + 1 / edge.m, edge.m, i])
        y_min = y_min + 1

    if len(border_points) == 0:
        for i, edge in enumerate(edges):
            if edge.active:
                border_points.append([edge.y_min[0], edge.m, i])

    for y in range(y_min, y_max + 1):
        border_points = sorted(border_points, key=lambda x: x[0])

        # find color in border scanline points
        color_A = interpolate_vectors(edges[border_points[0][2]].vertices[0], \
                                      edges[border_points[0][2]].vertices[1], \
                                      vcolors[border_points[0][2]], \
                                      vcolors[(border_points[0][2] + 1) % 3], \
                                      y, 2)
        color_B = interpolate_vectors(edges[border_points[1][2]].vertices[0], \
                                      edges[border_points[1][2]].vertices[1], \
                                      vcolors[border_points[1][2]], \
                                      vcolors[(border_points[1][2] + 1) % 3], \
                                      y, 2)

        for x in range(math.floor(border_points[0][0] + 0.5), \
                       math.floor(border_points[1][0] + 0.5) + 1):
            # find color in scanline
            updatedcanvas[y, x] = interpolate_vectors(np.array([math.floor(border_points[0][0] + 0.5), y]), \
                                               np.array([math.floor(border_points[1][0] + 0.5) + 1, y]), \
                                               color_A, color_B, x, 1)

        if y == y_max:
            break

        for point in border_points:
            point[0] = point[0] + 1 / point[1]

        for i, edge in enumerate(edges):
            if edge.y_min[1] == y + 1:
                edge.active = True
                actives = actives + 1
                border_points.append([edge.y_min[0], edge.m, i])

        if actives == 3:
            ## only one edge case for m = 0 on last edge
            if border_points[-1][1] == 0:
                del border_points[-1]
                continue
            for i, edge in enumerate(edges):
                if edge.y_max == y + 1:
                    if border_points[0][2] == i:
                        del border_points[0]
                    else:
                        del border_points[1]
                    edge.active = False
                    actives = actives - 1
                    break


    return updatedcanvas


# paints object described by args
def render(verts2d, faces, vcolors, depth, M, N):
    # calculate depth of each triagle
    triangles_depth = np.array(np.mean(depth[faces], axis = 1))

    # sort faces triangles depth
    indices = np.flip(np.argsort(triangles_depth))
    triangles_depth = triangles_depth[indices]
    faces = faces[indices]

    img = np.ones((M, N, 3))

    for face in faces:
        img = Gourauds(img, verts2d[face], vcolors[face])

    return img


# renders object described by args
def RenderObject(p3d, faces, vcolors, H, W, Rows, Columns, f, cv, K, cup):
    # project points
    p2d, depth = CameraLookingAt(f, cv, K - cv, cup, p3d)
    # get pixel coords
    n2d = rasterize(p2d, Rows, Columns, H, W)

    n2d = n2d.T

    # paint object 
    I = render(n2d, faces, vcolors, depth, Rows, Columns)

    return I

