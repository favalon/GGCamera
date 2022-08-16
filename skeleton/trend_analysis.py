import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from general.save_load import LoadBasic, SaveBasic
from skeleton.analysis_support import important_point_selection
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


def select_target_point(points_speed):
    max_value = 0
    max_point_index = 0
    max_point = ""
    for i, key in enumerate(points_speed):
        point_speed = points_speed[key][0]
        point_max = np.max(point_speed)
        if point_max > max_value:
            max_point = key
            max_value = point_max
            max_point_index = i

    return max_point, max_value, max_point_index


def plot_graph(points_speed, axis=None, act_name='', only_peak=True, show_plot=True):
    fig = plt.figure(figsize=(16, 7))
    if only_peak:
        point_name, peak_value, peak_i = select_target_point(points_speed)
        print("Action *{}* Peak Motion with Point at *{}*".format(act_name, point_name))
        plt.subplot(1, 1, 1)
        plt.title(f'{point_name} Axis {axis}', y=1.08)
        plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=None, hspace=0.5)
        point_speed = points_speed[point_name]
        plt.plot([i for i in range(point_speed[0].shape[0] - 2)], point_speed[0][:-2])
        plt.scatter(point_speed[1], point_speed[0][point_speed[1]], c='g')
        plt.scatter(point_speed[2], point_speed[0][point_speed[2]], c='r')
    else:
        for i, key in enumerate(points_speed):
            plt.subplot(3, 2, i+1)
            plt.title(f'{key} Axis {axis}', y=1.08)

            plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=None, hspace=0.5)

            point_speed = points_speed[key]

            plt.plot([i for i in range(point_speed[0].shape[0]-2)], point_speed[0][:-2])
            plt.scatter(point_speed[1], point_speed[0][point_speed[1]], c='g')
            plt.scatter(point_speed[2], point_speed[0][point_speed[2]], c='r')

    plt.suptitle('{} Change Curve'.format(act_name), fontsize=20)

    plt.show()
    return


def get_single_acceleration(skeleton_sequence, names=None, axis='x', act_name='', plot=False):

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
            point_max = np.max(point_speed)
            peaks, _ = find_peaks(point_speed, distance=30, height=(0.5 * point_max, point_max + 0.01))
            point_max = np.max(-point_speed)
            valley, _ = find_peaks(-point_speed, distance=30,  height=(-point_max - 0.01, 0.5 * -point_max))
            points_speed[point_name] = (point_speed, peaks, valley)

    else:
        for i in selected_index:
            point_name = names[i]
            point_position = skeleton_sequence[:, i]
            point_speed = np.diff(point_position) / np.diff([i for i in range(len(skeleton_sequence))])
            point_max = np.max(point_speed)
            peaks, _ = find_peaks(point_speed, distance=30, height=(0.5 * point_max, point_max + 0.01))
            valley, _ = find_peaks(-point_speed, distance=30, height=(-point_max - 0.01, 0.5 * -point_max))
            points_speed[point_name] = (point_speed, peaks, valley)

    if len(peaks) == 0:
        peaks = np.append(peaks, [0])
    if len(valley) == 0:
        valley = np.append(valley, [0])

    points_imp = important_point_selection(peaks, valley)

    if 0 not in points_imp:
        points_imp.insert(0, 0)
    if skeleton_sequence.shape[0]-1 not in points_imp:
        points_imp.append(skeleton_sequence.shape[0]-1)

    point_name, peak_value, peak_i = select_target_point(points_speed)

    if plot:
        plot_graph(points_speed, axis=axis, act_name=act_name)

    return points_speed, point_name, points_imp


def get_accelerations(skeleton_sequence, point_names, act_name='', mix_only="y"):

    diff_x = max(skeleton_sequence[0, 0, :, 0]) - min(skeleton_sequence[0, 0, :, 0])
    diff_y = max(skeleton_sequence[0, 0, :, 2]) - min(skeleton_sequence[0, 0, :, 2])
    diff_z = max(skeleton_sequence[0, 0, :, 1]) - min(skeleton_sequence[0, 0, :, 1])

    max_diff = max(diff_x, diff_y, diff_z)

    if mix_only == 'mix':
        mix_speed, point_name, points_imp = get_single_acceleration(skeleton_sequence[:, 0, :, :], names=point_names, axis='mix', act_name=act_name)
        return mix_speed, point_name, points_imp, max_diff
    else:
        if mix_only == 'x':
            x_speed, point_name, points_imp = get_single_acceleration(skeleton_sequence[:, 0, :, 0], names=point_names, axis='x', act_name=act_name)
            return x_speed, point_name, points_imp, max_diff
        if mix_only == 'y':
            y_speed, point_name, points_imp = get_single_acceleration(skeleton_sequence[:, 0, :, 2], names=point_names, axis='y', act_name=act_name)

            return y_speed, point_name, points_imp, max_diff
        if mix_only == 'z':
            z_speed, point_name, points_imp = get_single_acceleration(skeleton_sequence[:, 0, :, 1], names=point_names, axis='z', act_name=act_name)
            return z_speed, point_name, points_imp, max_diff


def process_action(skeleton_sequence, action_name=''):
    skeleton_sequence, point_names = get_skeleton_array(skeleton_sequence)

    mix_speed, point_name, points_imp, max_diff = get_accelerations(skeleton_sequence, point_names, act_name=action_name)

    return mix_speed, point_name, points_imp, max_diff


def decompose_action(path='local_data/skeleton', fn='action_standard_set1.json'):
    actions_set = LoadBasic.load_basic(fn=fn, path=path, file_type='json')

    for action in actions_set:
        action_name = action['ActionName']
        actions_seq = action['ActionData']

        process_action(actions_seq, action_name=action_name)


def split_store_action(path='local_data/skeleton', fn='action_standard_set1.json'):
    actions_set = LoadBasic.load_basic(fn=fn, path=path, file_type='json')

    for action in actions_set:
        action_name = action['ActionName']
        actions_seq = action['ActionData']

        SaveBasic.save_json(actions_seq, path=os.path.join(path, 'split_action'), fn=action_name+'.json')


if __name__ == '__main__':
    split_store_action(path='local_data/skeleton', fn='action_standard_set1.json')
    # decompose_action()