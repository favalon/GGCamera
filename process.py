import matplotlib.pyplot as plt
import numpy as np
from utils.cube import plotCubeAt
from utils.line import line_vector
from utils.line import rotation as rotation_line


def generate_spindle_torus(r, R, theta, phi, n=20):
    theta = np.linspace(theta[0] * np.pi, theta[1] * np.pi, n)
    phi = np.linspace(phi[0] * np.pi, phi[1] * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    return [x, y, z]


def rotate_torus(torus, rotates=(0, 0, 0)):
    t = np.transpose(np.array([torus[0], torus[1], torus[2]]), (1, 2, 0))

    x, y, z = np.transpose(np.dot(t, rotates), (2, 0, 1))

    return [x, y, z]


def scale_torus(torus, scales=(1, 1, 1)):
    t = np.transpose(np.array([torus[0], torus[1], torus[2]]), (1, 2, 0))
    scale = [[scales[0], 0, 0], [0, scales[1], 0], [0, 0, scales[2]]]
    x, y, z = np.transpose(np.dot(t, scale), (2, 0, 1))
    return [x, y, z]


def scale(point, scales):
    point = np.dot(point, np.array([[scales[0], 0, 0], [0, scales[1], 0], [0, 0, scales[2]]]))

    return point


def generate_event(r, R, rotates, scales, size=(0.5, 0.5, 0.5), dist_offset=None):
    colors = np.random.rand(2, 3)
    a = np.sqrt(np.sum(np.square(scales)))
    h = 2 * np.sqrt(np.square(r) - np.square(R))
    cross_h_1 = - h / 2
    cross_h_2 = h / 2

    cross_point_1 = np.array([0, cross_h_1 * scales[2] * 2, 0])
    cross_point_2 = np.array([0, cross_h_2 * scales[2] * 2, 0])

    cross_point_1 = cross_point_1#  - np.asarray(size) / 2
    cross_point_2 = cross_point_2#  - np.asarray(size) / 2

    cross_point_1 = np.dot(rotates, cross_point_1) + dist_offset
    cross_point_2 = np.dot(rotates, cross_point_2) + dist_offset

    event_1 = {'position': cross_point_1, 'size': (0.5, 0.5, 0.5), 'color': colors[0]}
    event_2 = {'position': cross_point_2, 'size': (0.5, 0.5, 0.5), 'color': colors[1]}

    return event_1, event_2


def plt_torus(ax, torus, event_1=None, event_2=None, points=None, cameras=None, focus=None, lim=3):
    if ax:
        ax1 = ax
    else:
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim([-lim, lim])
    ax1.set_ylim([-lim, lim])
    ax1.set_zlim([-lim, lim])
    # ax1.plot_surface(torus[0], torus[1], torus[2], linewidth=0, antialiased=False, shade=True, alpha=0.2)
    if event_1 and event_2:
        # pc = plotCubeAt([event_1['position'], event_2['position']],
        #                 sizes=[event_1['size'], event_2['size']],
        #                 colors=[event_1['color'], event_2['color']], edgecolor="k")
        ax1.scatter(event_1['position'][0], event_1['position'][1], event_1['position'][2], marker='o', c='y', s=80)
        ax1.scatter(event_2['position'][0], event_2['position'][1], event_2['position'][2], marker='o', c='b', s=80)

    # if points:
    #     for i, point in enumerate(points):
    #         if i % 30 == 0:
    #             ax1.scatter(point[0], point[1], point[2], marker='o', c='r', s=20)
    #
    # if focus:
    #     for i, point in enumerate(focus):
    #         if i % 30 == 0:
    #             ax1.scatter(point[0], point[1], point[2], marker='o', c='g', s=20)

    points = np.array(points) - focus

    ax1.plot(points[:, 0], points[:, 1], points[:, 2], '.r-', linewidth=2)

    # if cameras:
    #     for i, camera in enumerate(cameras):
    #         camera_shooting = np.array([points[i], focus[i]])
    #         # camera_shooting = np.array([points[i] - focus[i], [0, 0, 0]])
    #         if i % 10 == 0:
    #             ax1.plot(camera_shooting[:, 0], camera_shooting[:, 1], camera_shooting[:, 2], '.y:', linewidth=3)

    plt.show()


def calculate_point_projection(R, r, thetas, phis, sample=10, scales=None, rotates=None, dist_offset=None):
    thetas = np.linspace(thetas[0], thetas[1], sample)
    phis = np.linspace(phis[0], phis[1], sample)

    points = []
    for i in range(sample):
        theta = thetas[i]
        phi = phis[i]
        x = (R + r * np.cos(theta)) * np.cos(phi) * scales[0]
        y = (R + r * np.cos(theta)) * np.sin(phi) * scales[2]
        z = r * np.sin(theta) * scales[1]

        point = [x, z, y]
        # point = scale(point, scales)
        point = np.dot(rotates, point) + dist_offset
        points.append(point)

    return points


def camera_line_simulation(e1_pos_env, e2_pos_env, e1_pos_unit, e2_pos_unit, camera_poss, theta_start=-120,
                           theta_end=120, sample=49):
    # calculate the rotation any scale use to rotation unit event line to env event line
    eline_env, scale_env = line_vector(e1_pos_env, e2_pos_env)
    eline_unit, scale_unit = line_vector(e1_pos_unit, e2_pos_unit)

    rm_proj, scale_proj = rotation_line(eline_unit, eline_env, scale_env, scale_unit)

    focus_center_x = [e1_pos_unit[0] + (e2_pos_unit[0] - e1_pos_unit[0]) / (sample -1) * i for i in range(sample)]
    focus_center_y = [e1_pos_unit[1] + (e2_pos_unit[1] - e1_pos_unit[1]) / (sample -1) * i for i in range(sample)]
    focus_center_z = [e1_pos_unit[2] + (e2_pos_unit[2] - e1_pos_unit[2]) / (sample -1) * i for i in range(sample)]

    focus = []
    for i in range(sample):
        focus_center = [focus_center_x[i], focus_center_y[i], focus_center_z[i]]
        focus.append(focus_center)

    # e_center_pos_unit = e2_pos_unit[[]]

    angles = []
    direct_vectors = []

    for i, camera_pos in enumerate(camera_poss):
        camera_focus = focus[i]
        angle, direct_vector = camera_shot_angle(np.array(camera_pos), np.array(camera_focus))
        angles.append(angle)
        direct_vectors.append(direct_vector)

    return direct_vectors, angles, focus


def camera_shot_angle(camera_pos, focus_pos):
    d_v, dist = line_vector(camera_pos, focus_pos)

    s = np.linalg.norm(d_v)
    theta_1 = np.rad2deg(np.arccos(d_v[0] / s))
    theta_2 = np.rad2deg(np.arccos(d_v[1] / s))
    theta_3 = np.rad2deg(np.arccos(d_v[2] / s))

    return [theta_1, theta_2, theta_3], d_v


# def main():
#     # basic parameters
#     r = 1.2
#     R = 1
#
#     rotates = np.deg2rad((0, 90, 0))
#     scales = (1.5, 1, 1)
#
#     projection_test_theta = np.deg2rad((-120, 120))
#     projection_test_phi = np.deg2rad((-90, -90))
#
#     # generate the Spindle Torus
#     torus = generate_spindle_torus(r=r, R=R, theta=[0, 2], phi=[0, 2], n=20)
#
#     # torus transformation
#     torus = rotate_torus(torus, rotates=rotates)
#     torus = scale_torus(torus, scales=scales)
#
#     event_1, event_2 = generate_event(r, R, rotates=rotates,scales=scales,
#                                       size=(0.2 * scales[0], 0.2 * scales[1], 0.2 * scales[2]))
#
#     # projection
#     points = calculate_point_projection(R, r, projection_test_theta, projection_test_phi, rotates=rotates,
#                                         scales=scales)
#
#     direct_vectors, angles, focus = camera_line_simulation(event_2['position'], event_1['position'],
#                                                     event_2['position'], event_1['position'],
#                                                     points,
#                                                     theta_start=-120, theta_end=120)
#
#     plt_torus(torus, event_1=event_1, event_2=event_2, focus=focus, points=points, cameras=direct_vectors)

#
# if __name__ == "__main__":
#     main()
