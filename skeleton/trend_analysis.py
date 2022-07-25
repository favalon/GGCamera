import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from camera.output_cameras import output_cameras_track
import pandas as pd
import os
from general.save_load import LoadBasic, SaveBasic
from utils.line import line_vector, rotation


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
    point_names = {}

    for frame, skeleton in enumerate(skeleton_sequence):
        for i, points in enumerate(skeleton['Skeleton']):

            if frame == 0:
                point_name = points['Name']
                point_names[i] = point_name

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

    return all_skeleton_xyz, point_names


def plot_graph(points_speed, axis=None, act_name=''):
    row = 3
    col = 2
    fig = plt.figure(figsize=(16, 7))

    for i, key in enumerate(points_speed):
        plt.subplot(row, col, i+1)
        plt.title(f'{key} Axis {axis}', y=1.08)

        plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=None, hspace=0.5)

        point_speed = points_speed[key]

        plt.plot([i for i in range(point_speed.shape[0]-2)], point_speed[:-2])

    plt.suptitle('{} Change Curve'.format(act_name), fontsize=20)

    plt.show()
    return


def get_single_acceleration(skeleton_sequence, names=None, axis='x', act_name=''):

    points_speed = {}
    selected_index = [4, 8, 15, 23, 17]

    if axis == "mix":
        for i in selected_index:
            point_name = names[i]
            point_position = skeleton_sequence[:, i]

            x_diff = np.diff(point_position[:, 0])
            y_diff = np.diff(point_position[:, 2])
            z_diff = np.diff(point_position[:, 1])

            mix_diff = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

            point_speed = mix_diff / np.diff([i for i in range(len(skeleton_sequence))])
            points_speed[point_name] = point_speed

    else:
        for i in selected_index:
            point_name = names[i]
            point_position = skeleton_sequence[:, i]
            point_speed = np.diff(point_position) / np.diff([i for i in range(len(skeleton_sequence))])
            points_speed[point_name] = point_speed


    plot_graph(points_speed, axis=axis, act_name=act_name)

    return 0


def get_accelerations(skeleton_sequence, point_names, act_name='', mix_only=True):
    if mix_only:
        mix_acc = get_single_acceleration(skeleton_sequence[:, 0, :, :], names=point_names, axis='mix', act_name=act_name)
    else:
        mix_acc = get_single_acceleration(skeleton_sequence[:, 0, :, :], names=point_names, axis='mix',
                                          act_name=act_name)
        x_acc = get_single_acceleration(skeleton_sequence[:, 0, :, 0], names=point_names, axis='x', act_name=act_name)
        y_acc = get_single_acceleration(skeleton_sequence[:, 0, :, 2], names=point_names, axis='y', act_name=act_name)
        z_acc = get_single_acceleration(skeleton_sequence[:, 0, :, 1], names=point_names, axis='z', act_name=act_name)


def process_action(skeleton_sequence, action_name=''):
    skeleton_sequence, point_names = get_skeleton_array(skeleton_sequence)

    get_accelerations(skeleton_sequence, point_names, act_name=action_name)


def decompose_action(path='local_data/skeleton', fn='action_standard_set1.json'):
    actions_set = LoadBasic.load_basic(fn=fn, path=path, file_type='json')

    for action in actions_set:
        action_name = action['ActionName']
        actions_seq = action['ActionData']

        process_action(actions_seq, action_name=action_name)


    # centroid_start = skeleton_sequence[0, 2, :, :].mean(axis=0)
    #
    # centroid_end, dist_max, frame_max = get_moving_max(centroid_start, skeleton_sequence)
    #
    # uv1, dist1 = line_vector(centroid_end, centroid_start)

    # return centroid_end, centroid_start


if __name__ == '__main__':
    decompose_action()