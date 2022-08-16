import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from camera.output_cameras import output_cameras_track
import pandas as pd
import os
from general.save_load import LoadBasic, SaveBasic
from utils.line import line_vector, rotation
from process import generate_spindle_torus, generate_event, \
    rotate_torus, scale_torus, calculate_point_projection, camera_line_simulation, plt_torus
from skeleton.trend_analysis import process_action
from skeleton.analysis_support import get_all_focus_imp


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3)


def rotation_by_direction(base_line, target_line):
    uv1, dist1 = line_vector(base_line[:, 0], base_line[:, 1])
    uv2, dist2 = line_vector(target_line[:, 0], target_line[:, 1])
    rotation_matrix, scale = rotation(uv1, uv2, dist1, dist2)

    target_line_center = (target_line[:, 1] + target_line[:, 0])/2
    base_line_center = (np.array([0, 0, 0]))/2
    center_dist = target_line_center - base_line_center

    return rotation_matrix, scale, center_dist


def rotate_direction(points, rotation_matrix):

    # rotation_matrix, scale = rotation_by_direction(base_line, target_line)

    for i in range(points.shape[0]):
        points[i][0:3] = np.dot(rotation_matrix, np.array([points[i][0], points[i][1], points[i][2]]))

    return points


def move_direction(points, dist, moving_increment, start_point):
    # start_point 初始位置

    # centroid_current
    z_offset = points.mean(axis=0)

    z_offset[2] += 0.5

    for i in range(points.shape[0]):
        points[i] += start_point + dist * moving_increment - z_offset
    return points


def combine_rotation(rotates, rotation_matrix=None):
    # m_x = np.array([[1, 0, 0],
    #                 [0, np.cos(rotates[0]), -np.sin(rotates[0])],
    #                 [0, np.sin(rotates[0]), np.cos(rotates[0])]])
    #
    # m_y = np.array([[np.cos(rotates[1]), 0, np.sin(rotates[1])],
    #                 [0, 1, 0],
    #                 [-np.sin(rotates[1]), 0, np.cos(rotates[1])]])
    #
    # m_z = np.array([[np.cos(rotates[2]), -np.sin(rotates[2]), 0],
    #                 [np.sin(rotates[2]), np.cos(rotates[2]), 0],
    #                 [0, 0, 1]])
    #
    # if rotation_matrix is not None:
    #
    #     return np.dot(np.dot(np.dot(m_x, m_y), m_z), rotation_matrix)
    # else:
    #     return np.dot(np.dot(m_x, m_y), m_z)
    return np.dot(rotation_matrix, rotates)


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, path='local_data/skeleton', fn='test_data.json', target_fn='test_data.json'):
        self.path = path
        self.fn = fn
        self.stream = self.data_stream()

        # for test roration
        self.frames = 10
        self.focus_points, self.focus_frames, self.focus_speed, self.focus_seq, self.focus_side, \
            self.theta_ratio = decompose_action(path=path, fn=fn)
        self.base_line = np.array([[0, 0], [0, 1], [0, 0]], dtype=np.float32)
        # self.target_line = np.concatenate((np.reshape(self.centroid_start, (3, 1)),
        #                                    np.reshape(self.centroid_end, (3, 1))), axis=1)

        # import target position
        self.target_focus_point = get_target_focus_point(path=path, fn=target_fn)

        self.target_line = np.concatenate((np.reshape(self.focus_points[0], (3, 1)),
                                            np.reshape(self.target_focus_point, (3, 1))), axis=1)

        self.target_uv, self.target_dist = line_vector(self.target_line[:, 0], self.target_line[:, 1])
        self.action_dist = self.target_dist
        self.rotation_matrix, self.target_dist, self.base2target_dist \
            = rotation_by_direction(self.base_line, self.target_line)

        # Setup the figure and axes...
        self.cameras_points = [[0, 0, 0]]
        self.focus = [[0, 0, 0]]
        self.cameras_rotation = [[0, 0, 0]]
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=40, init_func=self.setup_plot, blit=False)

    def setup_plot(self, lim=5):
        """Initial drawing of the scatter plot."""

        data, point, focus, rotate = next(self.stream)

        x, y, z = data.T
        # line = np.array([[0, 1], [0, 1], [0, 0]])
        self.scat = self.ax.scatter(x, y, z, c='y', s=20)

        self.create_torus()

        # self.ax.scatter(self.centroid_end[0], self.centroid_end[1], self.centroid_end[2])
        # self.ax.scatter(self.centroid_start[0], self.centroid_start[1], self.centroid_start[2])
        self.ax.plot(self.base_line[0, :], self.base_line[1, :], self.base_line[2, :], '.r-', linewidth=2)
        self.ax.plot(self.target_line[0, :], self.target_line[1, :], self.target_line[2, :], '.b-', linewidth=3)
        self.ax.plot(self.focus_seq[:, 0, 0], self.focus_seq[:, 0, 1], self.focus_seq[:, 0, 2], 'g-', linewidth=3)
        # k_line = self.base_line.copy()
        # k_line[:, 0] = np.dot(self.rotation_matrix, k_line[:, 0])
        # k_line[:, 1] = np.dot(self.rotation_matrix, k_line[:, 1])
        # self.ax.plot(k_line[0, :], k_line[1, :], k_line[2, :], '.r-', linewidth=2)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([-lim, lim])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        skeleton_sequence = LoadBasic.load_basic(fn=self.fn, path=self.path, file_type='json')
        self.frames = len(skeleton_sequence)
        while True:
            for frame, skeleton in enumerate(skeleton_sequence):
                all_skeleton_xyz = np.zeros((4, len(skeleton['Skeleton']), 3))
                for i, points in enumerate(skeleton['Skeleton']):
                    all_skeleton_xyz[0, i, 0] = points['LocalPosition']['x']
                    all_skeleton_xyz[0, i, 2] = points['LocalPosition']['y']
                    all_skeleton_xyz[0, i, 1] = points['LocalPosition']['z']

                    all_skeleton_xyz[1, i, 0] = points['LocalRotation']['x']
                    all_skeleton_xyz[1, i, 1] = points['LocalRotation']['y']
                    all_skeleton_xyz[1, i, 2] = points['LocalRotation']['z']

                    all_skeleton_xyz[2, i, 0] = points['WorldPosition']['x']
                    all_skeleton_xyz[2, i, 2] = points['WorldPosition']['y']
                    all_skeleton_xyz[2, i, 1] = points['WorldPosition']['z']

                    all_skeleton_xyz[3, i, 0] = points['WorldRotation']['x']
                    all_skeleton_xyz[3, i, 1] = points['WorldRotation']['y']
                    all_skeleton_xyz[3, i, 2] = points['WorldRotation']['z']

                s, c = np.ones((len(skeleton['Skeleton']), 2)).T * 30
                tmp = np.c_[all_skeleton_xyz[0, :, 0],
                            all_skeleton_xyz[0, :, 1],
                            all_skeleton_xyz[0, :, 2]]

                tmp = rotate_direction(tmp, self.rotation_matrix)

                moving_increment = 0 # frame/len(skeleton_sequence) * self.action_dist

                move_direction(tmp, self.target_uv, moving_increment, self.target_line[:, 0])

                point, focus, rotate = self.cameras_points[frame], self.focus[frame], self.cameras_rotation[frame]

                yield tmp, point, focus, rotate

    def update(self, i):
        """Update the scatter plot."""
        data, point, focus, rotate = next(self.stream)
        camera_shooting = np.array([point + focus])

        data = np.concatenate((data, camera_shooting), axis=0)

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # Set x and y data...
        self.scat._offsets3d = (x, y, z)
        self.scat.set_array(data[:, 2])
        self.ax.set_title("rotate theta1: {}, theta2: {}, theta3: {}".format(int(rotate[0]), int(rotate[1]), int(rotate[2])))
        # Set sizes...
        # self.scat.set_sizes(data[:, 3])
        # # Set colors..
        # self.scat.set_array(data[:, 4])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        # time.sleep(0.1)
        return self.scat,

    def create_torus(self):
        # basic parameters
        surface_scale = 0.1
        r = 2 * surface_scale
        R = 1 * surface_scale
        h = 2 * np.sqrt(np.square(r) - np.square(R))
        all_scale = self.target_dist/h/2

        rotates = Rotation.from_rotvec([np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]).as_matrix()
        scales = (1 * all_scale, 3 * all_scale, 1 * all_scale)

        projection_test_theta = np.deg2rad((-180 * self.theta_ratio, 180 * self.theta_ratio))
        if self.focus_side == 'left':
            projection_test_phi = np.deg2rad((180, 180))
        else:
            projection_test_phi = np.deg2rad((0, 0))

        # generate the Spindle Torus
        torus = generate_spindle_torus(r=r, R=R, theta=[0, 2], phi=[0, 2], n=20)

        rt = Rotation.from_matrix(self.rotation_matrix)
        rt_a = rt.as_euler('xyz', degrees=True)
        print(rt_a)

        # torus transformation
        torus = scale_torus(torus, scales=scales)
        # torus = rotate_torus(torus, rotates=rotates)
        # torus = scale_torus(torus, scales=scales)

        event_1, event_2 = generate_event(r, R, rotates=np.eye(3),
                                          dist_offset=self.base2target_dist, scales=scales)

        # projection
        points = calculate_point_projection(R, r, projection_test_theta, projection_test_phi,
                                            self.focus_points, self.focus_frames, self.focus_speed, self.focus_seq,
                                            sample=self.frames, rotates=np.eye(3),
                                            dist_offset=self.base2target_dist, scales=scales)

        direct_vectors, angles, focus = camera_line_simulation(event_1['position'], event_2['position'],
                                                               event_1['position'], event_2['position'],
                                                               points, sample=self.frames,
                                                               theta_start=projection_test_theta[0],
                                                               theta_end=projection_test_theta[1],
                                                               given_focus=self.focus_seq[:, 0, :])

        output = output_cameras_track(points, focus, angles, frame_offset=301, focus_side=self.focus_side)

        SaveBasic.save_json(data=output, path=os.path.join(self.path), fn="camera_test.json")

        self.cameras_points = points
        self.focus = focus
        self.cameras_rotation = angles
        plt_torus(self.ax, torus, event_1=event_1, event_2=event_2, focus=focus, points=points, cameras=direct_vectors)


def get_moving_max(centroid_start, skeleton_sequence, centroid=False):
    dist_max = 0
    frame_max = 0
    centroid_max = centroid_start

    for frame in range(skeleton_sequence.shape[0]):
        centroid_current = skeleton_sequence[frame, 0, :, :].mean(axis=0)
        dist = np.linalg.norm(centroid_current - centroid_start)
        if dist > dist_max:
            dist_max = dist
            frame_max = frame
            centroid_max = centroid_current

    return centroid_max, dist_max, frame_max


def get_moving_min(centroid_start, skeleton_sequence):
    dist_min = 9999
    frame_min = 0
    centroid_min = centroid_start

    for frame in range(skeleton_sequence.shape[0]):
        centroid_current = skeleton_sequence[frame, 0, :, :].mean(axis=0)
        dist = np.linalg.norm(centroid_current - centroid_start)
        if dist < dist_min:
            dist_min = dist
            frame_min = frame
            centroid_min = centroid_current

    return centroid_min, dist_min, frame_min


def get_skeleton_array(skeleton_sequence, centroid="centroid"):

    total_frames = len(skeleton_sequence)
    if centroid == 'centroid':
        all_skeleton_xyz = np.zeros((total_frames, 4, len(skeleton_sequence[0]['Skeleton']), 3))
    else:
        all_skeleton_xyz = np.zeros((total_frames, 4, 1, 3))
    for frame, skeleton in enumerate(skeleton_sequence):
        if centroid == 'centroid':
            for i, points in enumerate(skeleton['Skeleton']):
                all_skeleton_xyz[frame, 0, i, 0] = points['LocalPosition']['x']
                all_skeleton_xyz[frame, 0, i, 2] = points['LocalPosition']['y']
                all_skeleton_xyz[frame, 0, i, 1] = points['LocalPosition']['z']

                all_skeleton_xyz[frame, 1, i, 0] = points['LocalRotation']['x']
                all_skeleton_xyz[frame, 1, i, 1] = points['LocalRotation']['y']
                all_skeleton_xyz[frame, 1, i, 2] = points['LocalRotation']['z']

                all_skeleton_xyz[frame, 2, i, 0] = points['WorldPosition']['x']
                all_skeleton_xyz[frame, 2, i, 2] = points['WorldPosition']['y']
                all_skeleton_xyz[frame, 2, i, 1] = points['WorldPosition']['z']

                all_skeleton_xyz[frame, 3, i, 0] = points['WorldRotation']['x']
                all_skeleton_xyz[frame, 3, i, 1] = points['WorldRotation']['y']
                all_skeleton_xyz[frame, 3, i, 2] = points['WorldRotation']['z']
        else:
            for i, points in enumerate(skeleton['Skeleton']):
                if points["Name"] == centroid:
                    all_skeleton_xyz[frame, 0, 0, 0] = points['LocalPosition']['x']
                    all_skeleton_xyz[frame, 0, 0, 2] = points['LocalPosition']['y']
                    all_skeleton_xyz[frame, 0, 0, 1] = points['LocalPosition']['z']

                    all_skeleton_xyz[frame, 1, 0, 0] = points['LocalRotation']['x']
                    all_skeleton_xyz[frame, 1, 0, 1] = points['LocalRotation']['y']
                    all_skeleton_xyz[frame, 1, 0, 2] = points['LocalRotation']['z']

                    all_skeleton_xyz[frame, 2, 0, 0] = points['WorldPosition']['x']
                    all_skeleton_xyz[frame, 2, 0, 2] = points['WorldPosition']['y']
                    all_skeleton_xyz[frame, 2, 0, 1] = points['WorldPosition']['z']

                    all_skeleton_xyz[frame, 3, 0, 0] = points['WorldRotation']['x']
                    all_skeleton_xyz[frame, 3, 0, 1] = points['WorldRotation']['y']
                    all_skeleton_xyz[frame, 3, 0, 2] = points['WorldRotation']['z']

    return all_skeleton_xyz


def decompose_action(path='local_data/skeleton', fn='test_data.json', use_centroid=False):
    skeleton_sequence = LoadBasic.load_basic(fn=fn, path=path, file_type='json')

    # skeleton_sequence = get_skeleton_array(skeleton_sequence, centroid=False)

    # 1. get focus point:
    if use_centroid:
        skeleton_sequence = get_skeleton_array(skeleton_sequence, centroid="centroid")
        focus_start = skeleton_sequence[0, 0, :, :].mean(axis=0)

        focus_max, dist_max, frame_max = get_moving_max(focus_start, skeleton_sequence)

        focus_min, dist_min, frame_min = get_moving_min(focus_start, skeleton_sequence)

        focus_all = [focus_min, focus_max]
        frames_imp = [0, frame_max]
        focus_side = 'right'

    else:
        mix_speed, point_name, frames_imp, max_diff = process_action(skeleton_sequence, action_name=fn.split('.')[0])

        skeleton_sequence = get_skeleton_array(skeleton_sequence, centroid=point_name)

        focus_start = skeleton_sequence[0, 0, :, :].mean(axis=0)

        focus_max, dist_max, frame_max = get_moving_max(focus_start, skeleton_sequence)

        focus_min, dist_min, frame_min = get_moving_min(focus_start, skeleton_sequence)

        focus_all = get_all_focus_imp(skeleton_sequence, frames_imp)

        if 'L' in point_name:
            focus_side = 'left'
        else:
            focus_side = 'right'

        # theta moving ratio
        diff_x = np.max(skeleton_sequence[:, 0, :, 0]) - np.min(skeleton_sequence[:, 0, :, 0])
        diff_y = np.max(skeleton_sequence[0, 0, :, 2]) - np.min(skeleton_sequence[0, 0, :, 2])
        diff_z = np.max(skeleton_sequence[0, 0, :, 1]) - np.min(skeleton_sequence[0, 0, :, 1])

        focus_max_diff = max(diff_x, diff_y, diff_z)
        theta_ratio = abs(focus_max_diff/max_diff)
    return focus_all, frames_imp, mix_speed[point_name], skeleton_sequence[:, 0, :, :], focus_side, theta_ratio


def get_target_focus_point(path='local_data/skeleton', fn='test_data.json', use_start=False):
    skeleton_sequence = LoadBasic.load_basic(fn=fn, path=path, file_type='json')
    skeleton_sequence = get_skeleton_array(skeleton_sequence, centroid="centroid")
    if use_start:
        focus_point = skeleton_sequence[0, 2, :, :].mean(axis=0)
    else:
        focus_point = skeleton_sequence[-1, 2, :, :].mean(axis=0)

    return focus_point


if __name__ == '__main__':
    a = AnimatedScatter(path='local_data/skeleton', fn='actor_data.json', target_fn='target_data.json')
    plt.show()

    # decompose_action()
