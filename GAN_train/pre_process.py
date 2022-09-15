import os
import numpy as np
from general.save_load import LoadBasic, SaveBasic
from skeleton.main import get_skeleton_array


def prep_for_region(body_region, pose):

    # index 0-head
    head = np.take(pose, [16, 17, 18, 19], axis=1)
    body_region[:, 0, :4, :] = head
    # index 1-torsor
    torsor = np.take(pose, [0, 9, 10, 11, 24], axis=1)
    body_region[:, 1, :5, :] = torsor
    # index 2.1-left arm
    larm = np.take(pose, [12, 13, 14, 15], axis=1)
    body_region[:, 2, 0:4, :] = larm
    # index 2.2-3-right arm
    rarm = np.take(pose, [20, 21, 22, 23], axis=1)
    body_region[:, 3, 0:4, :] = rarm
    # index 3.1-4-left leg
    lleg = np.take(pose, [0, 1, 2, 3, 4], axis=1)
    body_region[:, 4, :5, :] = lleg
    # index 3.2-5-right leg
    rleg = np.take(pose, [0, 5, 6, 7, 8], axis=1)
    body_region[:, 5, :5, :] = rleg

    return body_region


def pre_process_single_action_data(action_data, camera, action_name="default", intensity=1, distance=2):
    # 1. raw data
    skeleton_sequence = get_skeleton_array(action_data)

    body_region_pos = np.zeros((skeleton_sequence.shape[0], 6, 5, 3))
    prep_for_region(body_region_pos, skeleton_sequence[:, 0, :, :])

    # 2. raw v
    # 3. raw diff
    # 4. raw split partial
    # 5. camera initial + different
    # 6. intensity
    # 7. target distance
    print(1)


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