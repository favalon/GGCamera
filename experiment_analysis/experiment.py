import math
import os
import matplotlib.pyplot as plt
import numpy as np
from skeleton.trend_analysis import process_action, plot_graph
from general.save_load import LoadBasic
from scipy.signal import savgol_filter
import scipy.spatial.distance as ssd
import scipy.stats as ss

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


def exp2_curve_compare_plot(grad_generate_seq, real_generate_seq, relate_generate_seq, point_speed, axis=""):

    smooth_window = 15

    # smooth the line
    noise = np.random.normal(point_speed, 1) * 0.0001
    grad_generate_seq = savgol_filter(grad_generate_seq + noise, smooth_window, 3)
    real_generate_seq = savgol_filter(real_generate_seq + noise, smooth_window, 3)
    relate_generate_seq = savgol_filter(relate_generate_seq + noise, smooth_window, 3)
    point_speed = savgol_filter(point_speed + noise, smooth_window, 3)

    point_speed = -point_speed

    # fig = plt.figure(figsize=(16, 7))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # plt.title(f'{point_name} Axis {axis}', y=1.08)
    # ax1.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=None, hspace=0.5)
    ax2.plot([i for i in range(point_speed.shape[0])], point_speed, label="action")
    ax1.plot([i for i in range(point_speed.shape[0])], grad_generate_seq, marker="v", c="r", label="generate")
    ax1.plot([i for i in range(point_speed.shape[0])], real_generate_seq, marker="o", c="g", label="real")
    ax1.plot([i for i in range(point_speed.shape[0])], relate_generate_seq, marker="1", c="y", label="relate")

    # figure info
    plt.title(f'Action Vs Camera Movement Speed Change ({axis} axis)')
    plt.xlabel("frames")
    ax1.set_ylabel('distance change per frame (camera)', color='black')
    ax2.set_ylabel('distance change per frame (action)', color='black')
    ax1.legend()
    ax2.legend(loc='upper center')
    plt.show()


    print(f'Similarity: Real Vs Skeleton Action ({axis} axis)')
    print(f'correlation distance {ssd.correlation(real_generate_seq, point_speed)}')
    # print(np.correlate(real_generate_seq, abs(point_speed), mode='valid'))
    # print(np.corrcoef(real_generate_seq, abs(point_speed)))
    print(f'correlation distance {ss.spearmanr(real_generate_seq, point_speed)}')

    print(f'Similarity: Generate Vs Skeleton Action ({axis} axis)')
    print(f'correlation distance {ssd.correlation(grad_generate_seq, point_speed)}')
    # print(np.correlate(grad_generate_seq, abs(point_speed), mode='valid'))
    # print(np.corrcoef(grad_generate_seq, abs(point_speed)))
    print(f'correlation distance {ss.spearmanr(grad_generate_seq, point_speed)}')

    print(f'Similarity: Ref Vs Skeleton Action ({axis} axis)')
    print(f'correlation distance {ssd.correlation(relate_generate_seq, point_speed)}')
    # print(np.correlate(relate_generate_seq, abs(point_speed), mode='valid'))
    # print(np.corrcoef(relate_generate_seq, abs(point_speed)))
    print(f'correlation distance {ss.spearmanr(relate_generate_seq, point_speed)}')

    return 0


def exp2_seq_dis_diff(generate_seq, real_seq, relate_seq, actions_seq, axis="x", max_frame=100):

    axis2index = {"x": 0, "y": 1, "z": 2}

    # get seq
    generate_select_seq = generate_seq[:, 0, axis2index[axis]]
    real_select_seq = real_seq[:, 0, axis2index[axis]]
    relate_select_seq = relate_seq[:, 0, axis2index[axis]]

    # get gradient

    grad_generate_seq = np.gradient(generate_select_seq)
    real_generate_seq = np.gradient(real_select_seq)
    relate_generate_seq = np.gradient(relate_select_seq)

    speed, name, imp, diff = process_action(actions_seq, action_name="kneel_front_prostrate_1", plot=False, mix_only=axis)
    point_speed = plot_graph(speed, axis=axis, act_name="kneel_front_prostrate_1", only_peak=True, show_plot=False)

    exp2_curve_compare_plot(grad_generate_seq, real_generate_seq, relate_generate_seq,
                            point_speed[0][10:max_frame], axis=axis)

    return 0


def exp3_curve_compare_plot(generate_seq_02, generate_seq_04, generate_seq_06,
                            generate_seq_08, generate_seq_1,
                            real_generate_seq, relate_generate_seq, point_speed, axis=""):

    smooth_window = 15

    # smooth the line
    noise = np.random.normal(point_speed, 1) * 0.0001
    generate_seq_02 = savgol_filter(generate_seq_02 + noise, smooth_window, 3)
    generate_seq_04 = savgol_filter(generate_seq_04 + noise, smooth_window, 3)
    generate_seq_06 = savgol_filter(generate_seq_06 + noise, smooth_window, 3)
    generate_seq_08 = savgol_filter(generate_seq_08 + noise, smooth_window, 3)
    generate_seq_1 = savgol_filter(generate_seq_1 + noise, smooth_window, 3)

    real_generate_seq = savgol_filter(real_generate_seq + noise, smooth_window, 3)
    relate_generate_seq = savgol_filter(relate_generate_seq + noise, smooth_window, 3)
    point_speed = savgol_filter(point_speed + noise, smooth_window, 3)

    point_speed = -point_speed

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax2.plot([i for i in range(point_speed.shape[0])], point_speed, label="action")
    ax1.plot([i for i in range(point_speed.shape[0])], generate_seq_02, marker="v", c="r", label="0.2")
    ax1.plot([i for i in range(point_speed.shape[0])], generate_seq_04, marker="v", c="r", label="0.4")
    ax1.plot([i for i in range(point_speed.shape[0])], generate_seq_06, marker="v", c="r", label="0.6")
    ax1.plot([i for i in range(point_speed.shape[0])], generate_seq_08, marker="v", c="r", label="0.8")
    ax1.plot([i for i in range(point_speed.shape[0])], generate_seq_1, marker="v", c="r", label="1")

    ax1.plot([i for i in range(point_speed.shape[0])], real_generate_seq, marker="o", c="g", label="real")
    ax1.plot([i for i in range(point_speed.shape[0])], relate_generate_seq, marker="1", c="y", label="relate")

    ax1.legend()
    # ax2.legend(loc='upper center')
    plt.show()
    return 0


def exp3_emotional_action(path, generate_seq, real_seq, relate_seq, actions_seq, axis="x", max_frame=80):
    generate_seq_02 = reformat_seq(
        LoadBasic.load_basic("camera_2_int_0.2.json", path=os.path.join(path, "intentsity"), file_type="json")['data'])[10:max_frame, :, :]
    generate_seq_04 = reformat_seq(
        LoadBasic.load_basic("camera_2_int_0.4.json", path=os.path.join(path, "intentsity"), file_type="json")['data'])[10:max_frame, :, :]
    generate_seq_06 = reformat_seq(
        LoadBasic.load_basic("camera_2_int_0.6.json", path=os.path.join(path, "intentsity"), file_type="json")['data'])[10:max_frame, :, :]
    generate_seq_08 = reformat_seq(
        LoadBasic.load_basic("camera_2_int_0.8.json", path=os.path.join(path, "intentsity"), file_type="json")['data'])[10:max_frame, :, :]
    generate_seq_1 = reformat_seq(
        LoadBasic.load_basic("camera_2_int_1.json", path=os.path.join(path, "intentsity"), file_type="json")['data'])[10:max_frame, :, :]


    axis2index = {"x": 0, "y": 1, "z": 2}

    # get seq
    generate_seq_02 = generate_seq_02[:, 0, axis2index[axis]]
    generate_seq_04 = generate_seq_04[:, 0, axis2index[axis]]
    generate_seq_06 = generate_seq_06[:, 0, axis2index[axis]]
    generate_seq_08 = generate_seq_08[:, 0, axis2index[axis]]
    generate_seq_1 = generate_seq_1[:, 0, axis2index[axis]]
    real_select_seq = real_seq[:, 0, axis2index[axis]]
    relate_select_seq = relate_seq[:, 0, axis2index[axis]]

    speed, name, imp, diff = process_action(actions_seq, action_name="kneel_front_prostrate_1", plot=False,
                                            mix_only=axis)
    point_speed = plot_graph(speed, axis=axis, act_name="kneel_front_prostrate_1", only_peak=True, show_plot=False)


    exp3_curve_compare_plot(generate_seq_02, generate_seq_04, generate_seq_06,
                            generate_seq_08, generate_seq_1,
                            real_select_seq, relate_select_seq, point_speed[0][10:max_frame], axis=axis)

    return 0


def main(path, max_frame=90):
    generate_seq = reformat_seq(LoadBasic.load_basic("camera_seq_test.json", path=path, file_type="json"))
    real_seq = reformat_seq(LoadBasic.load_basic("camera_seq_test_real.json", path=path, file_type="json"))
    relate_seq = reformat_seq(LoadBasic.load_basic("camera_seq_test_relate.json", path=path, file_type="json"))
    action_seq = LoadBasic.load_basic("kneel_front_prostrate_1.json",
                                      path=os.path.join("GGCamera_data", "action_raw"), file_type="json")
    # exp1_seq_dis_diff(generate_seq[:max_frame, :, :], real_seq[:max_frame, :, :], relate_seq[:max_frame, :, :])

    # exp2_seq_dis_diff(generate_seq[10:max_frame, :, :], real_seq[10:max_frame, :, :], relate_seq[10:max_frame, :, :], action_seq, axis="x", max_frame=max_frame)

    exp3_emotional_action(path,
                          generate_seq[10:max_frame, :, :],
                          real_seq[10:max_frame, :, :],
                          relate_seq[10:max_frame, :, :],
                          action_seq, axis="z", max_frame=max_frame)

if __name__ == '__main__':
    data_path = "GGCamera_data/camera_output"
    main(data_path)
