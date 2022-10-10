import math
import os
import random
import copy
from Const import norm_M1, rand5 as rands1
import torch
import numpy as np
import glob
from general.save_load import LoadBasic

body_masks_valid = np.array(
    [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]).reshape(
    (1, 6, 5))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_skeleton_array(skeleton_sequence, max_frame=100):
    all_skeleton_xyz = np.zeros((max_frame, 26, 4, 3))
    for frame, skeleton in enumerate(skeleton_sequence):
        if frame >= max_frame:
            break
        for i, points in enumerate(skeleton['Skeleton']):
            all_skeleton_xyz[frame, i, 0, 0] = points['LocalPosition']['x']
            all_skeleton_xyz[frame, i, 0, 2] = points['LocalPosition']['y']
            all_skeleton_xyz[frame, i, 0, 1] = points['LocalPosition']['z']

            all_skeleton_xyz[frame, i, 1, 0] = points['LocalRotation']['x']
            all_skeleton_xyz[frame, i, 1, 1] = points['LocalRotation']['y']
            all_skeleton_xyz[frame, i, 1, 2] = points['LocalRotation']['z']

            all_skeleton_xyz[frame, i, 2, 0] = points['WorldPosition']['x']
            all_skeleton_xyz[frame, i, 2, 2] = points['WorldPosition']['y']
            all_skeleton_xyz[frame, i, 2, 1] = points['WorldPosition']['z']

            all_skeleton_xyz[frame, i, 3, 0] = points['WorldRotation']['x']
            all_skeleton_xyz[frame, i, 3, 1] = points['WorldRotation']['y']
            all_skeleton_xyz[frame, i, 3, 2] = points['WorldRotation']['z']

    return all_skeleton_xyz[:, :, 0:2, :]


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


def compute_topo_test(actinfo):
    poses = actinfo[0]
    #  compute topo based on plv
    plv = poses[:, 1, 2, :]
    plv = plv.reshape(plv.shape[0], 1, 1, plv.shape[1])
    diff = poses - plv
    diff = np.square(diff)
    diff = np.sum(diff, axis=3)
    diff = np.sqrt(diff)
    diff = diff * body_masks_valid
    return diff


def get_skeleton_diff(skeleton_sequence, position=0, norm=False):
    skeleton_diff = np.abs(skeleton_sequence[1:, :, position, :] - skeleton_sequence[:-1, :, position, :])
    if norm:
        normal = np.linalg.norm(skeleton_diff)
        skeleton_diff = skeleton_diff / normal
    addition = np.copy(skeleton_diff[-1])
    addition = np.expand_dims(addition, axis=0)
    skeleton_diff = np.concatenate((skeleton_diff, addition), axis=0)
    return skeleton_diff


def get_skeleton_speed(skeleton_sequence, position=0, norm=False):
    skeleton_speed = skeleton_sequence[1:, :, position, :] - skeleton_sequence[:-1, :, position, :]
    if norm:
        normal = np.linalg.norm(skeleton_speed)
        skeleton_speed = skeleton_speed / normal

    addition = np.copy(skeleton_speed[-1])
    addition = np.expand_dims(addition, axis=0)
    skeleton_speed = np.concatenate((skeleton_speed, addition), axis=0)

    return skeleton_speed


def enhance_act_feature(act_info):
    act_info = np.expand_dims(act_info, axis=0)

    actsnorm = norm_M1[:, :-12]
    toponorm = norm_M1[:, -12:-2]
    actsnorm = np.split(actsnorm, 12, axis=1)

    acts_diff = np.abs(act_info[:, 1:, :, :30] - act_info[:, :-1, :, :30])
    acts_topo = act_info[:, :, :, 30:]
    rot_diff = acts_diff[:, :, :, 15:]

    acts_v = act_info[:, 1:, :, :30] - act_info[:, :-1, :, :30]
    rot_v = acts_v[:, :, :, 15:]

    action_data = act_info[:, :, :, :30]
    action_data[:, :, :, 15:] = np.deg2rad(action_data[:, :, :, 15:])

    logi = (rot_diff >= 180).astype(np.int32)
    if (np.sum(logi) != 0):
        logi_nega = (rot_v >= 0).astype(np.int32)
        logi_nega = 2 * logi_nega * (-1) + 1
        rot_v = rot_v + logi * logi_nega * 360

    logi = (rot_diff >= 180).astype(np.int32)
    if (np.sum(logi) != 0):
        logi_scale = 2 * logi * (-1) + 1
        logi_bais = logi * 360
        rot_diff = rot_diff * logi_scale + logi_bais

    acts_diff[:, :, :, 15:] = np.deg2rad(rot_diff)
    acts_v[:, :, :, 15:] = np.deg2rad(rot_v)

    np.seterr(divide="ignore", invalid="ignore")
    # acts_diff[:, :, :, :15] = (acts_diff[:, :, :, :15] - actsnorm[1]) / actsnorm[0]
    # acts_diff[:, :, :, 15:] = (acts_diff[:, :, :, 15:] - actsnorm[3]) / actsnorm[2]
    #
    # acts_v[:, :, :, :15] = (acts_v[:, :, :, :15] - actsnorm[5]) / actsnorm[4]
    # acts_v[:, :, :, 15:] = (acts_v[:, :, :, 15:] - actsnorm[7]) / actsnorm[6]
    #
    # action_data[:, :, :, :15] = (action_data[:, :, :, :15] - actsnorm[9]) / actsnorm[8]
    # action_data[:, :, :, 15:] = (action_data[:, :, :, 15:] - actsnorm[11]) / actsnorm[10]
    #
    # acts_topo = (acts_topo - toponorm[:, 5:]) / toponorm[:, :5]
    action_data = np.concatenate((action_data, acts_topo), axis=3)
    #
    acts_topo = (acts_topo[:, 1:, :, :] + acts_topo[:, :-1, :, :]) / 2
    acts_diff = np.concatenate((acts_diff, acts_topo), axis=3)
    acts_v = np.concatenate((acts_v, acts_topo), axis=3)

    acts_diff = np.nan_to_num(acts_diff, nan=0.00001)
    acts_v = np.nan_to_num(acts_v, nan=0.00001)
    action_data = np.nan_to_num(action_data, nan=0.00001)

    acts_diff = torch.from_numpy(acts_diff).float()
    acts_v = torch.from_numpy(acts_v).float()
    action_data = torch.from_numpy(action_data).float().numpy()

    rand_noise_1 = torch.from_numpy(copy.deepcopy(rands1))
    rand_noise_1 = rand_noise_1.float()

    rand_noise_2 = torch.from_numpy(copy.deepcopy(rands1))
    rand_noise_2 = rand_noise_2.float()

    acts_v = torch.cat((rand_noise_1, acts_v), dim=1).numpy()
    acts_diff = torch.cat((rand_noise_2, acts_diff), dim=1).numpy()

    return action_data, acts_diff, acts_v


def format_camera_data(camera_data, max_frame=100):
    camera_data_format = np.zeros((len(camera_data), 2, 3))
    for frame, data in enumerate(camera_data):
        camera_data_format[frame, 0, 0] = data["LocalPosition"]['x']
        camera_data_format[frame, 0, 1] = data["LocalPosition"]['y']
        camera_data_format[frame, 0, 2] = data["LocalPosition"]['z']

        camera_data_format[frame, 1, 0] = data["LocalRotation"]['x']
        camera_data_format[frame, 1, 1] = data["LocalRotation"]['y']
        camera_data_format[frame, 1, 2] = data["LocalRotation"]['z']

    # camera_data_diff = np.zeros((len(camera_data), 2, 3))
    # camera_data_diff[1:, :, :] = np.diff(camera_data_format, axis=0)
    #
    # camera_data_format = np.nan_to_num(camera_data_format, nan=0.00001)
    # camera_data_diff = np.nan_to_num(camera_data_diff, nan=0.00001)
    #
    # camera_data_format = np.expand_dims(camera_data_format, axis=0)
    # camera_data_diff = np.expand_dims(camera_data_diff, axis=0)
    #
    # camera_data = np.concatenate((camera_data_format, camera_data_diff), axis=0)
    camera_data_reformat = np.zeros((max_frame, 2, 3))
    if len(camera_data) <= max_frame:
        camera_data_reformat[:len(camera_data)] = camera_data_format
    else:
        camera_data_reformat = camera_data_format[:100, :, :]

    return camera_data_reformat


def random_sign():
    return random.randint(0, 1) * 2 - 1


def random_position_generation(position, distance):
    x = random_sign() * round(random.random() * 10, 2)
    y = random_sign() * round(random.random() * 10, 2)

    x_addition = round(random.random() * distance, 2)
    y_addition = math.sqrt(distance ** 2 - x_addition ** 2)

    # actor_pos
    position[:, 0, 0] = x
    position[:, 0, 1] = y
    # target_pos
    position[:, 1, 0] = x + random_sign() * x_addition
    position[:, 1, 1] = y + random_sign() * y_addition

    return position


def pre_process_single_action_data(action_data, camera_data,
                                   action_name="default",
                                   intensity=1.0, distance=2, max_frame=100,
                                   norm=True):
    # 1. raw data
    skeleton_sequence = get_skeleton_array(action_data, max_frame=max_frame)
    # 2. raw diff
    diff_skeleton = get_skeleton_diff(skeleton_sequence, norm=norm)
    # 3. raw v
    speed_skeleton = get_skeleton_speed(skeleton_sequence, norm=norm)
    # 4. raw split partial
    body_region_pos = np.zeros((skeleton_sequence.shape[0], 6, 5, 3))
    body_region_rot = np.zeros((skeleton_sequence.shape[0], 6, 5, 3))
    body_region_pos = prep_for_region(body_region_pos, skeleton_sequence[:, :, 0, :])
    body_region_rot = prep_for_region(body_region_rot, skeleton_sequence[:, :, 1, :])

    body_region_pos = np.expand_dims(body_region_pos, axis=0)
    body_region_rot = np.expand_dims(body_region_rot, axis=0)

    act_info = np.concatenate((body_region_pos,  body_region_rot), axis=0)

    # topo_ori = compute_topo_test(body_region_pos[0, :, :, :, :])
    # topo_diff = compute_topo_test(body_region_pos[1, :, :, :, :])
    # topo_speed = compute_topo_test(body_region_pos[2, :, :, :, :])
    # topo_ori = np.nan_to_num(topo_ori, nan=0.00001)
    # topo_diff = np.nan_to_num(topo_diff, nan=0.00001)
    # topo_speed = np.nan_to_num(topo_speed, nan=0.00001)

    topo = compute_topo_test(act_info)

    body_region_pos = body_region_pos.reshape((body_region_pos.shape[0], body_region_pos.shape[1],
                                               body_region_pos.shape[2],
                                               body_region_pos.shape[3] * body_region_pos.shape[4]))

    poses = act_info[0]
    poses = poses.reshape((poses.shape[0], poses.shape[1], poses.shape[2] * poses.shape[3]))
    rots = act_info[1]
    rots = rots.reshape((rots.shape[0], rots.shape[1], rots.shape[2] * rots.shape[3]))

    act_info = np.concatenate((poses, rots, topo), axis=2)

    # enhance act info by diff and v
    act_info_base, act_info_diff, act_info_v = enhance_act_feature(act_info)

    # 5. camera initial theta (frames, theta)
    init_theta = np.zeros((max_frame, 1))
    init_theta[:, :] = math.radians(-80 / math.pi)

    # 6. intensity (frames, intensity)
    emo_intensity = np.zeros((max_frame, 1))
    emo_intensity[:, :] = intensity

    # 7. target distance (frames distance)
    dis_diff = np.zeros((max_frame, 1))
    dis_diff[:, :] = distance

    # 8. target and actor position shape(frame, 2, (x, y, z))
    position = np.zeros((max_frame, 2, 3))
    position = random_position_generation(position, int(distance))

    # 9. camera sequence
    camera_data = format_camera_data(camera_data["data"], max_frame=max_frame)

    # 3. init_camera
    init_camera = np.zeros((max_frame, 2, 3))
    init_camera = np.tile(camera_data[0], (max_frame, 1, 1))

    init_camera = np.nan_to_num(np.expand_dims(init_camera, axis=0), nan=0.00001)
    init_theta = np.nan_to_num(np.expand_dims(init_theta, axis=0), nan=0.00001)
    emo_intensity = np.nan_to_num(np.expand_dims(emo_intensity, axis=0), nan=0.00001)
    dis_diff = np.nan_to_num(np.expand_dims(dis_diff, axis=0), nan=0.00001)
    position = np.nan_to_num(np.expand_dims(position, axis=0), nan=0.00001)
    camera_data = np.nan_to_num(np.expand_dims(camera_data, axis=0), nan=0.00001)

    return act_info_base, act_info_diff, act_info_v, init_camera, \
           init_theta, emo_intensity, dis_diff, position, camera_data


def pre_process(path):
    action_raw = os.listdir(os.path.join(path, "action_raw"))
    for action in action_raw:
        action_raw_src = os.path.join(path, "action_raw", action)


def normalization_data(data):
    data_normalized = np.zeros(data.shape)
    data_normalized[:, :, :, :15] = (data[:, :, :, :15] - np.min(data[:, :, :, :15])) / np.ptp(data[:, :, :, :15])
    data_normalized[:, :, :, 15:30] = (data[:, :, :, 15:30] - np.min(data[:, :, :, 15:30])) / np.ptp(data[:, :, :, 15:30])
    data_normalized[:, :, :, 30:] = (data[:, :, :, 30:] - np.min(data[:, :, :, 30:])) / np.ptp(data[:, :, :, 30:])
    return data_normalized


if __name__ == '__main__':
    data_root = "GGCamera_data"

    total_files = glob.glob(os.path.join(data_root, "action2camera_tracks/*/*.json"))
    all_files_len = len(total_files)
    # test part

    # data split
    act_info_base_contain = np.zeros((all_files_len, 100, 6, 35))
    act_info_diff_contain = np.zeros((all_files_len, 100, 6, 35))
    act_info_v_contain = np.zeros((all_files_len, 100, 6, 35))
    init_camera_contain = np.zeros((all_files_len, 100,  2, 3))
    init_theta_contain = np.zeros((all_files_len, 100,  1))
    emo_intensity_contain = np.zeros((all_files_len, 100,  1))
    dis_diff_contain = np.zeros((all_files_len, 100, 1))
    position_contain = np.zeros((all_files_len, 100, 2, 3))
    camera_data_contain = np.zeros((all_files_len, 100, 2, 3))

    i = 0

    for file in glob.glob(os.path.join(data_root, "action_raw/*.json")):
        act_name = file.split("\\")[-1].split(".json")[0]

        for camera_file in glob.glob(os.path.join(data_root, "action2camera_tracks/{}/*.json".format(act_name))):

            print("processing: {}".format(camera_file))

            dis = camera_file.split("\\")[-1].split("_")[1]
            intensity = camera_file.split("\\")[-1].split("_")[3].split(".json")[0]

            action = LoadBasic.load_basic(fn="{}.json".format(act_name), path=os.path.join(data_root, "action_raw"), file_type='json')
            camera = LoadBasic.load_basic(fn="{}/camera_{}_int_{}.json".format(act_name, dis, intensity),
                                          path=os.path.join(data_root, "action2camera_tracks"), file_type='json')

            act_info_base, act_info_diff, act_info_v, init_camera, init_theta, emo_intensity, dis_diff, position, camera_data \
                = pre_process_single_action_data(action, camera, distance=dis, intensity=intensity)

            act_info_base_contain[i] = act_info_base
            act_info_diff_contain[i] = act_info_diff
            act_info_v_contain[i] = act_info_v
            init_camera_contain[i] = init_camera
            init_theta_contain[i] = init_theta
            emo_intensity_contain[i] = emo_intensity
            dis_diff_contain[i] = dis_diff
            position_contain[i] = position
            camera_data_contain[i] = camera_data

            i += 1
            print(i)


    act_info_base_contain = normalization_data(act_info_base_contain)
    act_info_diff_contain = normalization_data(act_info_diff_contain)
    act_info_v_contain = normalization_data(act_info_v_contain)

    np.save(os.path.join(data_root, "processed_np/act_info_base_contain.npy"), act_info_base_contain)
    np.save(os.path.join(data_root, "processed_np/act_info_diff_contain.npy"), act_info_diff_contain)
    np.save(os.path.join(data_root, "processed_np/act_info_v_contain.npy"), act_info_v_contain)
    np.save(os.path.join(data_root, "processed_np/init_camera_contain.npy"), init_camera_contain)
    np.save(os.path.join(data_root, "processed_np/init_theta_contain.npy"), init_theta_contain)
    np.save(os.path.join(data_root, "processed_np/emo_intensity_contain.npy"), emo_intensity_contain)
    np.save(os.path.join(data_root, "processed_np/dis_diff_contain.npy"), dis_diff_contain)
    np.save(os.path.join(data_root, "processed_np/position_contain.npy"), position_contain)
    np.save(os.path.join(data_root, "processed_np/camera_data_contain.npy"), camera_data_contain)

    print("data generation end")
