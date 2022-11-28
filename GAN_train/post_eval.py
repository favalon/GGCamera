import os

import numpy as np
import torch

from general.save_load import SaveBasic
from models.NetArchi_bc import Generator


def load_single_test(path, single_i=100):
    with open(os.path.join(path, 'act_info_base_contain.npy'), 'rb') as f:
        act_info_base_contain = np.load(f)
    with open(os.path.join(path, 'act_info_diff_contain.npy'), 'rb') as f:
        act_info_diff_contain = np.load(f)
    with open(os.path.join(path, 'act_info_v_contain.npy'), 'rb') as f:
        act_info_v_contain = np.load(f)
    with open(os.path.join(path, 'init_camera_contain.npy'), 'rb') as f:
        init_camera_contain = np.load(f)
        init_camera_contain = \
            init_camera_contain.reshape((init_camera_contain.shape[0], init_camera_contain.shape[1], 6))
    with open(os.path.join(path, 'init_theta_contain.npy'), 'rb') as f:
        init_theta_contain = np.load(f)
    with open(os.path.join(path, 'emo_intensity_contain.npy'), 'rb') as f:
        emo_intensity_contain = np.load(f)
    with open(os.path.join(path, 'dis_diff_contain.npy'), 'rb') as f:
        dis_diff_contain = np.load(f)
    with open(os.path.join(path, 'position_contain.npy'), 'rb') as f:
        position_contain = np.load(f)
        position_contain = \
            position_contain.reshape((position_contain.shape[0], position_contain.shape[1], 6))
    with open(os.path.join(path, 'camera_data_contain.npy'), 'rb') as f:
        camera_data_contain = np.load(f)
        camera_data_contain = \
            camera_data_contain.reshape((camera_data_contain.shape[0], camera_data_contain.shape[1], 6))

    train_data = torch.cat((torch.from_numpy(act_info_base_contain), torch.from_numpy(act_info_diff_contain),
                            torch.from_numpy(act_info_v_contain)), dim=3)

    train_label = torch.cat((torch.from_numpy(init_camera_contain), torch.from_numpy(init_theta_contain),
                             torch.from_numpy(emo_intensity_contain), torch.from_numpy(dis_diff_contain),
                             torch.from_numpy(position_contain), torch.from_numpy(camera_data_contain)), dim=2)

    return train_data[single_i], train_label[single_i]


def process_evaluation(generator, data1, data2, bs=10):
    data1 = data1.repeat((bs, 1, 1, 1))
    data2 = data2.repeat((bs, 1, 1))

    act_diff = data1[:, :, :, :35].float()
    action = data1[:, :, :, 35:70].float()
    act_v = data1[:, :, :, 70:].float()

    inicam = data2[:, :, 0:6].float()
    initheta = data2[:, :, 6:7].float()
    emoint = data2[:, :, 7:8].float()
    inipos = data2[:, :, 9:15].float()
    cam_real = data2[:, :, 15:21].float().numpy()

    if (torch.cuda.is_available()):
        model = generator.cuda()
        act_diff = act_diff.cuda()
        action = action.cuda()
        act_v = act_v.cuda()
        inicam = inicam.cuda()
        initheta = initheta.cuda()
        emoint = emoint.cuda()
        inipos = inipos.cuda()

    cam_seq = generator(act_diff, action, act_v, inipos, initheta, emoint, inicam)

    return cam_seq.cpu().detach().numpy(), cam_real


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_the_movement(cam_seq):
    cam_seq[:, 0] = smooth(cam_seq[:, 0], 15)
    cam_seq[:, 1] = smooth(cam_seq[:, 1], 15)
    cam_seq[:, 2] = smooth(cam_seq[:, 2], 15)
    return cam_seq


def reformat_output(cam_seq, offset=123, smooth=True):
    if smooth:
        cam_seq = smooth_the_movement(cam_seq)

    reformat_seq = []
    for frame in range(cam_seq.shape[0]):
        line = {
            "Frame": frame + offset,
            "FocusedName": "Flynn Rider",
            "LocalPosition": {
                "x": round(float(cam_seq[frame, 0]), 3),
                "y": round(float(cam_seq[frame, 1]), 3),
                "z": round(float(cam_seq[frame, 2]), 3)
            },
            "LocalRotation": {
                "x": round(float(cam_seq[frame, 3]), 2) if abs(float(cam_seq[frame, 3])) > 1 else 0,
                "y": round(float(cam_seq[frame, 4]), 2) if abs(float(cam_seq[frame, 4])) > 1 else 0,
                "z": round(float(cam_seq[frame, 5]), 2) if abs(float(cam_seq[frame, 5])) > 1 else 0
            },
            "ModelWorldPosition": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "ModelWorldRotation": {
                "x": 0.0,
                "y": 41.36733,
                "z": 0.0
            }
        }

        reformat_seq.append(line)

    return reformat_seq


def main(path):
    # 1. load test data
    test_data, test_label = load_single_test(os.path.join(path, "processed_np"), single_i=274)

    # 2. load Generator model
    model_path = os.path.join(path, "checkpointGAN", "w_g.1000.pt")
    G = Generator(10, 35, 6)
    G.load(model_path)

    # 3. evaluation
    can_seq, real_seq = process_evaluation(G, test_data, test_label)

    # 4. output value
    cam_seq = reformat_output(can_seq[0, :, :6], smooth=True)
    SaveBasic.save_json(data=cam_seq, path=os.path.join(path, 'camera_output'),
                        fn="camera_seq_test.json")
    real_seq = reformat_output(real_seq[0, :, :6], smooth=False)
    SaveBasic.save_json(data=real_seq, path=os.path.join(path, 'camera_output'),
                        fn="camera_seq_test_real.json")

    print("evaluation end")


if __name__ == '__main__':
    # data path
    data_path = "GGCamera_data"
    main(data_path)
