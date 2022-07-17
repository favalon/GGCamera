import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.cube import plotCubeAt, cuboid_data
from utils.line import line_vector
from utils.line import rotation as rotation_line
from utils.rotate_3d import Rotate


def generate_spindle_torus(r, R, theta, phi, n=20):
    theta = np.linspace(theta[0] * np.pi, theta[1] * np.pi, n)
    phi = np.linspace(phi[0] * np.pi, phi[1] * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    return [x, y, z]


def rotate_torus(torus, rotates=(0, 0, 0)):

    t = np.transpose(np.array([torus[0], torus[1], torus[2]]), (1,2,0))

    m_x = np.array([[1, 0, 0],
                    [0, np.cos(rotates[0]), -np.sin(rotates[0])],
                    [0, np.sin(rotates[0]), np.cos(rotates[0])]])

    m_y = np.array([[np.cos(rotates[1]), 0, np.sin(rotates[1])],
                    [0, 1, 0],
                    [-np.sin(rotates[1]), 0, np.cos(rotates[1])]])

    m_z = np.array([[np.cos(rotates[2]), -np.sin(rotates[2]), 0],
                    [np.sin(rotates[2]), np.cos(rotates[2]), 0],
                    [0, 0, 1]])

    x, y, z = np.transpose(np.dot(np.dot(np.dot(t, m_x), m_y), m_z), (2, 0, 1))

    return [x, y, z]


def scale_torus(torus, scales=(1, 1, 1)):
    t = np.transpose(np.array([torus[0], torus[1], torus[2]]), (1, 2, 0))
    scale = [[scales[0], 0, 0], [0, scales[1], 0], [0, 0, scales[2]]]
    x, y, z = np.transpose(np.dot(t, scale), (2, 0, 1))
    return [x, y, z]


def rotation(point, rotates):
    # point = Rotate(point, point1=[0, 0, 0], point2=[1, 0, 0], theta=np.deg2rad(rotates[0]))
    # point = Rotate(point, point1=[0, 0, 0], point2=[0, 1, 0], theta=np.deg2rad(rotates[1]))
    # point = Rotate(point, point1=[0, 0, 0], point2=[0, 0, 1], theta=np.deg2rad(rotates[2]))

    m_x = np.array([[1, 0, 0],
                    [0, np.cos(rotates[0]), -np.sin(rotates[0])],
                    [0, np.sin(rotates[0]), np.cos(rotates[0])]])

    m_y = np.array([[np.cos(rotates[1]), 0, np.sin(rotates[1])],
                    [0, 1, 0],
                    [-np.sin(rotates[1]), 0, np.cos(rotates[1])]])

    m_z = np.array([[np.cos(rotates[2]), -np.sin(rotates[2]), 0],
                    [np.sin(rotates[2]), np.cos(rotates[2]), 0],
                    [0, 0, 1]])

    # point = np.dot(np.dot(np.dot(point, m_x), m_y), m_z)
    point = np.dot(m_z, np.dot(m_y, np.dot(m_x, point)))
    return point


def scale(point, scales):

    point = np.dot(point, np.array([[scales[0], 0, 0], [0, scales[1], 0], [0, 0, scales[2]]]))

    return point


def generate_event(r, R, rotates, scales, size=(0.5, 0.5, 0.5)):
    colors = np.random.rand(2, 3)
    h = 2 * np.sqrt(np.square(r) - np.square(R))
    cross_h_1 = h / 2
    cross_h_2 = - h / 2

    cross_point_1 = np.array([0, 0, cross_h_1])
    cross_point_2 = np.array([0, 0, cross_h_2])

    cross_point_1 = rotation(cross_point_1, rotates)
    cross_point_2 = rotation(cross_point_2, rotates)
    #cross_point_1 = scale(cross_point_1, scales)
    #cross_point_2 = scale(cross_point_2, scales)

    # rotation
    # cross_point_1 = rotation(cross_point_1, rotates)
    # cross_point_2 = rotation(cross_point_2, rotates)

    cross_point_1 = cross_point_1 * np.asarray(scales) - np.asarray(size) / 2
    cross_point_2 = cross_point_2 * np.asarray(scales) - np.asarray(size) / 2

    event_1 = {'position': cross_point_1, 'size': size, 'color': colors[0]}
    event_2 = {'position': cross_point_2, 'size': size, 'color': colors[1]}

    return event_1, event_2


def plt_torus(torus, event_1=None, event_2=None, points=None, cameras=None, lim=3):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim([-lim, lim])
    ax1.set_ylim([-lim, lim])
    ax1.set_zlim([-lim, lim])
    ax1.plot_surface(torus[0], torus[1], torus[2], linewidth=0, antialiased=False, shade=True, alpha=0.5)
    if event_1 and event_2:
        pc = plotCubeAt([event_1['position'], event_2['position']],
                        sizes=[event_1['size'], event_2['size']],
                        colors=[event_1['color'], event_2['color']], edgecolor="k")
        ax1.add_collection3d(pc)

    if points:
        for point in points:
            ax1.scatter(point[0], point[1], point[2], marker='o', c='r', s=20)

    points = np.array(points)

    ax1.plot(points[:, 0], points[:, 1], points[:, 2], '.r-', linewidth=2)

    if cameras:
        for i, camera in enumerate(cameras):

            camera_shooting = np.array([points[i], camera * 5])

            ax1.plot(camera_shooting[:, 0], camera_shooting[:, 1], camera_shooting[:, 2], '.y:', linewidth=2)



    plt.show()


def calculate_point_projection(R, r, thetas, phis, sample=10, scales=None, rotates=None):

    thetas = np.linspace(thetas[0], thetas[1], sample)
    phis = np.linspace(phis[0], phis[1], sample)

    points = []
    for i in range(sample):
        theta = thetas[i]
        phi = phis[i]
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)

        point = [x, y, z]
        point = rotation(point, rotates)
        point = scale(point, scales)
        points.append(point)

    return points


def camera_line_simulation(e1_pos_env, e2_pos_env, e1_pos_unit, e2_pos_unit, camera_poss, theta_start=-120, theta_end=120):
    # calculate the rotation any scale use to rotation unit event line to env event line
    eline_env, scale_env = line_vector(e1_pos_env, e2_pos_env)
    eline_unit, scale_unit = line_vector(e1_pos_unit, e2_pos_unit)

    rm_proj, scale_proj = rotation_line(eline_unit, eline_env, scale_env, scale_unit)

    e_center_pos_unit = e2_pos_unit # [0, 0, 0]

    angles = []
    direct_vectors = []

    for camera_pos in camera_poss:
        angle, direct_vector = camera_shot_angle(np.array(camera_pos), np.array(e_center_pos_unit))
        angles.append(angle)
        direct_vectors.append(direct_vector)

    return direct_vectors, angles


def camera_shot_angle(camera_pos, focus_pos):

    d_v, dist = line_vector(camera_pos, focus_pos)
    s = np.linalg.norm(d_v)
    theta_1 = np.arccos(d_v[0]/s)
    theta_2 = np.arccos(d_v[1] / s)
    theta_3 = np.arccos(d_v[2] / s)

    return [theta_1, theta_2, theta_3], d_v


def main():
    # basic parameters
    r = 1.2
    R = 1

    rotates = np.deg2rad((0, 90, 0))
    scales = (1.5, 0.7, 1)

    projection_test_theta = np.deg2rad((-120, 120))
    projection_test_phi = np.deg2rad((-100, -100))

    # generate the Spindle Torus
    torus = generate_spindle_torus(r=r, R=R, theta=[0, 2], phi=[0, 2], n=20)

    # torus transformation
    torus = rotate_torus(torus, rotates=rotates)
    torus = scale_torus(torus, scales=scales)

    event_1, event_2 = generate_event(r, R, rotates=rotates, scales=scales,
                                      size=(0.2*scales[0], 0.2*scales[1], 0.2*scales[2]))

    # projection
    points = calculate_point_projection(R, r, projection_test_theta, projection_test_phi, rotates=rotates, scales=scales)

    direct_vectors, angles = camera_line_simulation(event_1['position'], event_2['position'],
                                                    event_1['position'], event_2['position'],
                                                    points,
                                                    theta_start=-120, theta_end=120)

    plt_torus(torus, event_1=event_1, event_2=event_2, points=points, cameras=direct_vectors)


if __name__ == "__main__":
    main()
