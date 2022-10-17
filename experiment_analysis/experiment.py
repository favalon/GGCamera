import math

import matplotlib.pyplot as plt
import numpy as np

from general.save_load import LoadBasic


def reformat_seq(cam_seq):
    cam_seq_format = np.zeros((len(cam_seq), 2, 3))
    for frame, cam_item in enumerate(cam_seq):
        cam_seq_format[frame, 0, 0] = cam_item["LocalPosition"]['x']
        cam_seq_format[frame, 0, 1] = cam_item["LocalPosition"]['y']
        cam_seq_format[frame, 0, 2] = cam_item["LocalPosition"]['z']

        cam_seq_format[frame, 1, 0] = math.radians(cam_item["LocalRotation"]['x'])
        cam_seq_format[frame, 1, 1] = math.radians(cam_item["LocalRotation"]['y'])
        cam_seq_format[frame, 1, 2] = math.radians(cam_item["LocalRotation"]['z'])

    return cam_seq_format


def dis_diff(generate_seq, real_seq, pc=(0, 0)):
    diff = []
    diff_acl = []
    for i in range(real_seq.shape[0]):
        diff_val = abs(generate_seq[i, pc[0], pc[1]] - real_seq[i, pc[0], pc[1]])
        if i == 0:
            diff_acl.append(diff_val)
        else:
            diff_acl.append(diff_acl[-1] + diff_val)
        diff.append(diff_val)

    return diff, diff_acl


def _plot(frames, generate_seq, real_seq, relate_seq, pc, row, col, plot_i):
    axis = ['X', 'Y', 'Z']
    atri = ['Distance', 'Rotation']

    diff_val, diff_acl_val = dis_diff(generate_seq, real_seq, pc=pc)
    diff_val_relate, diff_acl_val_relate = dis_diff(relate_seq, real_seq, pc=pc)

    plt.subplot(row, col, plot_i)
    plt.title(f'Camera Placement {atri[pc[0]]} Difference among {axis[pc[1]]} Axis', y=1.08)
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=None, hspace=0.5)
    # plt.plot(frames, diff_val, c='b', label="pair")
    plt.plot(frames, diff_acl_val, c='darkviolet', label="generate_acc")
    plt.plot(frames, diff_acl_val_relate, c='violet', label="relate_acc")
    plt.legend()

    plt.subplot(row, col, plot_i + 1)
    plt.title(f'Camera Placement: Generated VS Ground Truth', y=1.08)
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=None, hspace=0.5)
    # plt.plot(frames, diff_acl_x)
    plt.plot(frames, generate_seq[:, pc[0], pc[1]].tolist(), c='r', label="generate")
    plt.plot(frames, relate_seq[:, pc[0], pc[1]].tolist(), c='b', label="relate")
    plt.plot(frames, real_seq[:, pc[0], pc[1]].tolist(), c='g', label="real")
    plt.scatter(frames, generate_seq[:, pc[0], pc[1]].tolist(), c='r', s=4)
    plt.scatter(frames, relate_seq[:, pc[0], pc[1]].tolist(), c='g', s=4)
    plt.scatter(frames, real_seq[:, pc[0], pc[1]].tolist(), c='g', s=4)
    plt.legend()


def exp1_seq_dis_diff(generate_seq, real_seq, relate_seq):
    frames = [i for i in range(real_seq.shape[0])]

    # abs diff plot
    fig = plt.figure(1, figsize=(16, 7))
    # x
    _plot(frames, generate_seq, real_seq, relate_seq, pc=(0, 0), row=3, col=2, plot_i=1)
    # y
    _plot(frames, generate_seq, real_seq, relate_seq, pc=(0, 1), row=3, col=2, plot_i=3)
    # z
    _plot(frames, generate_seq, real_seq, relate_seq, pc=(0, 2), row=3, col=2, plot_i=5)
    plt.show()

    fig = plt.figure(2, figsize=(16, 7))
    # rx
    _plot(frames, generate_seq, real_seq, relate_seq, pc=(1, 0), row=3, col=2, plot_i=1)
    # ry
    _plot(frames, generate_seq, real_seq, relate_seq, pc=(1, 1), row=3, col=2, plot_i=3)
    # rz
    _plot(frames, generate_seq, real_seq, relate_seq, pc=(1, 2), row=3, col=2, plot_i=5)
    plt.show()

    return 0


def main(path, max_frame=90):
    generate_seq = reformat_seq(LoadBasic.load_basic("camera_seq_test.json", path=path, file_type="json"))
    real_seq = reformat_seq(LoadBasic.load_basic("camera_seq_test_real.json", path=path, file_type="json"))
    relate_seq = reformat_seq(LoadBasic.load_basic("camera_seq_test_relate.json", path=path, file_type="json"))
    exp1_seq_dis_diff(generate_seq[:max_frame, :, :], real_seq[:max_frame, :, :], relate_seq[:max_frame, :, :])


if __name__ == '__main__':
    data_path = "GGCamera_data/camera_output"
    main(data_path)
