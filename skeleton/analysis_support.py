import numpy as np


def important_point_selection(peak, valley):
    peak_cursor = 0
    valley_cursor = 0
    node = []

    while peak_cursor < len(peak) or valley_cursor < len(valley):

        if valley_cursor == len(valley):
            node.append([peak[peak_cursor], 1])
            peak_cursor += 1
        elif peak_cursor == len(peak):
            node.append([valley[valley_cursor], 1])
            valley_cursor += 1
        else:
            if peak[peak_cursor] < valley[valley_cursor]:
                current_min_index = peak[peak_cursor]
                node.append([peak[peak_cursor], 1])
                peak_cursor += 1

            else:
                current_min_index = valley[valley_cursor]
                node.append([valley[valley_cursor], 0])
                valley_cursor += 1

    selected_node = [node[0][0]]
    pre_flag = node[0][1]
    for i in node:
        cur_flag = i[1]
        if cur_flag != pre_flag:
            selected_node.append(i[0])
            pre_flag = cur_flag

    return selected_node


def get_all_focus_imp(skeleton_sequence, frames_imp_index):
    all_focus_imp_value = []
    for frame in range(skeleton_sequence.shape[0]):
        if frame in frames_imp_index:
            all_focus_imp_value.append(skeleton_sequence[frame, 0, :, :].mean(axis=0))

    return all_focus_imp_value


def map_skeleton_sequence_diff(skeleton_sequence, skeleton_sequence_ori, intensity=0.1, is_moving=False):

    # map x,y,z changing to a better focus point
    mapped_skeleton_sequence = np.zeros((skeleton_sequence.shape))
    mapped_skeleton_sequence[0, :, :, :] = skeleton_sequence_ori[0, :, :, :]

    x_diff = np.diff(skeleton_sequence[:, 2, 0, 0])
    y_diff = np.diff(skeleton_sequence[:, 2, 0, 1])
    z_diff = np.diff(skeleton_sequence[:, 2, 0, 2])

    x_diff_loc = np.diff(skeleton_sequence[:, 0, 0, 0])
    y_diff_loc = np.diff(skeleton_sequence[:, 0, 0, 1])
    z_diff_loc = np.diff(skeleton_sequence[:, 0, 0, 2])

    if is_moving:
        for i in range(x_diff.shape[0]):
            mapped_skeleton_sequence[i + 1, 2, 0, 0] = mapped_skeleton_sequence[i, 2, 0, 0] + x_diff[i]
            mapped_skeleton_sequence[i + 1, 2, 0, 1] = mapped_skeleton_sequence[i, 2, 0, 1] + y_diff[i]
            mapped_skeleton_sequence[i + 1, 2, 0, 2] = mapped_skeleton_sequence[i, 2, 0, 2] + z_diff[i]

            mapped_skeleton_sequence[i + 1, 0, 0, 0] = mapped_skeleton_sequence[i, 0, 0, 0] + x_diff_loc[i]
            mapped_skeleton_sequence[i + 1, 0, 0, 1] = mapped_skeleton_sequence[i, 0, 0, 1] + y_diff_loc[i]
            mapped_skeleton_sequence[i + 1, 0, 0, 2] = mapped_skeleton_sequence[i, 0, 0, 2] + z_diff_loc[i]
    else:
        for i in range(x_diff.shape[0]):
            mapped_skeleton_sequence[i + 1, 2, 0, 0] = mapped_skeleton_sequence[i, 2, 0, 0] + x_diff[i] * intensity
            mapped_skeleton_sequence[i + 1, 2, 0, 1] = mapped_skeleton_sequence[i, 2, 0, 1] + y_diff[i] * intensity
            mapped_skeleton_sequence[i + 1, 2, 0, 2] = mapped_skeleton_sequence[i, 2, 0, 2] + z_diff[i] * intensity

            mapped_skeleton_sequence[i + 1, 0, 0, 0] = mapped_skeleton_sequence[i, 0, 0, 0] + x_diff_loc[i] * intensity
            mapped_skeleton_sequence[i + 1, 0, 0, 1] = mapped_skeleton_sequence[i, 0, 0, 1] + y_diff_loc[i] * intensity
            mapped_skeleton_sequence[i + 1, 0, 0, 2] = mapped_skeleton_sequence[i, 0, 0, 2] + z_diff_loc[i] * intensity

    return mapped_skeleton_sequence