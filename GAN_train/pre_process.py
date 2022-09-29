import os

import numpy as np

from general.save_load import LoadBasic
from skeleton.main import get_skeleton_array

body_masks_valid = np.array(
    [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]).reshape(
    (1, 6, 5))


def prep_for_region(body_region, pose):
    # index 0-head
    head = np.take(pose, [16, 17, 18, 19], axis=1)
    body_region[:, 0, :4, :] = head
    # index 1-torsor
    torsor = np.take(pose, [0, 9, 10, 11, 24], axis=1)
    body_region[:, 1, :5, :] = torsor
    # index 2.1-left arm
    larm = np.take(pose, [12, 13, 14, 15], axis=1)
    body_region[:, 2, :4, :] = larm
    # index 2.2-3-right arm
    rarm = np.take(pose, [20, 21, 22, 23], axis=1)
    body_region[:, 3, :4, :] = rarm
    # index 3.1-4-left leg
    lleg = np.take(pose, [0, 1, 2, 3, 4], axis=1)
    body_region[:, 4, :5, :] = lleg
    # index 3.2-5-right leg
    rleg = np.take(pose, [0, 5, 6, 7, 8], axis=1)
    body_region[:, 5, :5, :] = rleg

    return body_region


def compute_topo_test(poses):
    # compu topo based on plv
    plv = poses[:, 1, 2, :]
    plv = plv.reshape(plv.shape[0], 1, 1, plv.shape[1])
    diff = poses - plv
    diff = np.square(diff)
    diff = np.sum(diff, axis=3)
    diff = np.sqrt(diff)
    diff = diff * body_masks_valid
    return diff


def get_skeleton_diff(skeleton_sequence, position=0, norm=False):
    skeleton_diff = np.abs(skeleton_sequence[1:, position, :, :] - skeleton_sequence[:-1, position, :, :])
    if norm:
        normal = np.linalg.norm(skeleton_diff)
        skeleton_diff = skeleton_diff / normal
    addition = np.copy(skeleton_diff[-1])
    addition = np.expand_dims(addition, axis=0)
    skeleton_diff = np.concatenate((skeleton_diff, addition), axis=0)
    return skeleton_diff


def get_skeleton_speed(skeleton_sequence, position=0, norm=False):
    skeleton_speed = skeleton_sequence[1:, position, :, :] - skeleton_sequence[:-1, position, :, :]
    if norm:
        normal = np.linalg.norm(skeleton_speed)
        skeleton_speed = skeleton_speed / normal

    addition = np.copy(skeleton_speed[-1])
    addition = np.expand_dims(addition, axis=0)
    skeleton_speed = np.concatenate((skeleton_speed, addition), axis=0)

    return skeleton_speed


def format_camera_data(camera_data):
    camera_data_format = np.zeros((len(camera_data), 2, 3))
    for frame, data in enumerate(camera_data):
        camera_data_format[frame, 0, 0] = data["LocalPosition"]['x']
        camera_data_format[frame, 0, 1] = data["LocalPosition"]['y']
        camera_data_format[frame, 0, 2] = data["LocalPosition"]['z']

        camera_data_format[frame, 1, 0] = data["LocalRotation"]['x']
        camera_data_format[frame, 1, 1] = data["LocalRotation"]['y']
        camera_data_format[frame, 1, 2] = data["LocalRotation"]['z']
        print(1)

    camera_data_diff = np.zeros((len(camera_data), 2, 3))
    camera_data_diff[1:, :, :] = np.diff(camera_data_format, axis=0)

    camera_data_format = np.nan_to_num(camera_data_format, nan=0.00001)
    camera_data_diff = np.nan_to_num(camera_data_diff, nan=0.00001)

    camera_data_format = np.expand_dims(camera_data_format, axis=0)
    camera_data_diff = np.expand_dims(camera_data_diff, axis=0)

    camera_data = np.concatenate((camera_data_format, camera_data_diff), axis=0)

    return camera_data


def pre_process_single_action_data(action_data, camera_data, action_name="default", intensity=1, distance=2, norm=True):
    # 1. raw data
    skeleton_sequence = get_skeleton_array(action_data)
    # 2. raw diff
    diff_skeleton = get_skeleton_diff(skeleton_sequence, norm=norm)
    # 3. raw v
    speed_skeleton = get_skeleton_speed(skeleton_sequence, norm=norm)
    # 4. raw split partial
    body_region_pos = np.zeros((3, skeleton_sequence.shape[0], 6, 5, 3))
    prep_for_region(body_region_pos[0, :, :, :, :], skeleton_sequence[:, 0, :, :])
    prep_for_region(body_region_pos[1, :, :, :, :], diff_skeleton[:, :, :])
    prep_for_region(body_region_pos[2, :, :, :, :], speed_skeleton[:, :, :])

    body_region_pos = np.nan_to_num(body_region_pos, nan=0.00001)

    topo_ori = compute_topo_test(body_region_pos[0, :, :, :, :])
    topo_diff = compute_topo_test(body_region_pos[1, :, :, :, :])
    topo_speed = compute_topo_test(body_region_pos[2, :, :, :, :])
    topo_ori = np.nan_to_num(topo_ori, nan=0.00001)
    topo_diff = np.nan_to_num(topo_diff, nan=0.00001)
    topo_speed = np.nan_to_num(topo_speed, nan=0.00001)

    body_region_pos = body_region_pos.reshape((body_region_pos.shape[0], body_region_pos.shape[1],
                                               body_region_pos.shape[2],
                                               body_region_pos.shape[3] * body_region_pos.shape[4]))

    skeleton_data_ori = np.concatenate((body_region_pos[0, :, :, :], topo_ori), axis=2)
    skeleton_data_diff = np.concatenate((body_region_pos[1, :, :, :], topo_diff), axis=2)
    skeleton_data_speed = np.concatenate((body_region_pos[2, :, :, :], topo_speed), axis=2)

    skeleton_data = np.stack((skeleton_data_ori, skeleton_data_diff, skeleton_data_speed))
    # 5. camera initial theta (frames, theta)
    camera_data = format_camera_data(camera_data["data"])

    # 6. intensity (frames, intensity)
    # 7. target distance (frames distance)
    # 8. target and actor position shape(frame, 2, (x, y, z))


    return skeleton_data, camera_data


def pre_process(path):
    action_raw = os.listdir(os.path.join(path, "action_raw"))
    for action in action_raw:
        action_raw_src = os.path.join(path, "action_raw", action)


if __name__ == '__main__':
    data_root = "local_data/skeleton"

    # test part
    dis = 2
    intensity = 0.2
    action = LoadBasic.load_basic(fn="actor_data.json", path=data_root, file_type='json')
    camera = LoadBasic.load_basic(fn="camera_output/camera_{}_int_{}.json".format(dis, intensity), path=data_root,
                                  file_type='json')
    pre_process_single_action_data(action, camera)

    print("test end")
