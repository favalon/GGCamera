import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from general.save_load import LoadBasic
from utils.line import line_vector, rotation


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

    return rotation_matrix, scale


def rotate_direction(points, base_line, target_line):

    rotation_matrix, scale = rotation_by_direction(base_line, target_line)

    for i in range(points.shape[0]):
        points[i][0:3] = np.dot(rotation_matrix, np.array([points[i][0], points[i][1], points[i][2]]))

    return points


def move_direction(points, dist, moving_increment, start_point):
    # start_point 初始位置

    # centroid_current
    z_offset = points.mean(axis=0)

    for i in range(points.shape[0]):
        points[i] += start_point + dist * moving_increment - z_offset
    return points


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, path='local_data/skeleton', fn='test_data.json'):
        self.path = path
        self.fn = fn
        self.stream = self.data_stream()

        # for test roration
        self.centroid_end, self.centroid_start = decompose_action()
        self.base_line = np.array([[0, 0], [0, 1], [0, 0]])
        self.target_line = np.array([[0, 0], [-1, 1], [0, 0]])
        # self.target_line = np.concatenate((np.reshape(self.centroid_start, (3, 1)), np.reshape(self.centroid_end, (3, 1))), axis=1)
        self.target_uv, self.target_dist = line_vector(self.target_line[:, 0], self.target_line[:, 1])
        self.action_dist = 2

        self.

        # Setup the figure and axes...
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=40, init_func=self.setup_plot, blit=False)

    def setup_plot(self, lim=5):
        """Initial drawing of the scatter plot."""
        x, y, z= next(self.stream).T
        # line = np.array([[0, 1], [0, 1], [0, 0]])
        self.scat = self.ax.scatter(x, y, z, c='y', s=20)

        self.ax.scatter(self.centroid_end[0], self.centroid_end[1], self.centroid_end[2])
        self.ax.scatter(self.centroid_start[0], self.centroid_start[1], self.centroid_start[2])
        self.ax.plot(self.base_line[0, :], self.base_line[1, :], self.base_line[2, :], '.r-', linewidth=2)
        self.ax.plot(self.target_line[0, :], self.target_line[1, :], self.target_line[2, :], '.g-', linewidth=2)

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

                tmp = rotate_direction(tmp, self.base_line, self.target_line)

                moving_increment = frame/len(skeleton_sequence) * self.action_dist

                move_direction(tmp, self.target_uv, moving_increment, self.target_line[:, 0])

                yield tmp

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # Set x and y data...
        self.scat._offsets3d = (x, y, z)
        # Set sizes...
        # self.scat.set_sizes(data[:, 3])
        # # Set colors..
        # self.scat.set_array(data[:, 4])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        # time.sleep(0.1)
        return self.scat,


def get_moving_max(centroid_start, skeleton_sequence):
    dist_max = 0
    frame_max = 0
    centroid_end = centroid_start

    for frame in range(skeleton_sequence.shape[0]):
        centroid_current = skeleton_sequence[frame, 2, :, :].mean(axis=0)
        dist = np.linalg.norm(centroid_current - centroid_start)
        if dist > dist_max:
            dist_max = dist
            frame_max = frame
            centroid_end = centroid_current

    return centroid_end, dist_max, frame_max


def get_skeleton_array(skeleton_sequence):

    total_frames = len(skeleton_sequence)
    all_skeleton_xyz = np.zeros((total_frames, 4, len(skeleton_sequence[0]['Skeleton']), 3))
    for frame, skeleton in enumerate(skeleton_sequence):
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

    return all_skeleton_xyz


def decompose_action(path='local_data/skeleton', fn='test_data.json'):
    skeleton_sequence = LoadBasic.load_basic(fn=fn, path=path, file_type='json')

    skeleton_sequence = get_skeleton_array(skeleton_sequence)

    centroid_start = skeleton_sequence[0, 2, :, :].mean(axis=0)

    centroid_end, dist_max, frame_max = get_moving_max(centroid_start, skeleton_sequence)

    uv1, dist1 = line_vector(centroid_end, centroid_start)

    return centroid_end, centroid_start


if __name__ == '__main__':
    a = AnimatedScatter(path='local_data/skeleton', fn='test_data.json')
    plt.show()

    decompose_action()
