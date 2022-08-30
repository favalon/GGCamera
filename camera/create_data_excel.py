import pandas as pd

from general.save_load import LoadBasic, SaveBasic

scenes = ['']
actors = ['man', 'woman']
objects = ['man']
actions = ['stand_front_push_4', 'run_front_jump_1', 'stand_front_jump_5', 'interact_pickUp_04', 'stand_front_climb_1',
           'stand_front_swing_4', 'stand_front_interact_2', 'kneel_1_prostrate_1', 'jump_left_fall_1',
           'stand_front_pull_5', 'stand_front_kick_4', 'stand_front_punch_4', 'stand_front_clap_4']
diss = ['2', '4', '8']
directions = ['0']
intensitys = ['0.2', '0.6', '1.0']

position = {
    "Name": "ModelPosition",
    "LocalPosition": {
        "x": 0,
        "y": 0,
        "z": 0
    },
    "LocalRotation": {
        "x": 0,
        "y": 0,
        "z": 0
    },
    "WorldPosition": {
        "x": 0,
        "y": 0,
        "z": 0
    },
    "WorldRotation": {
        "x": 0,
        "y": 0,
        "z": 0
    }
}


def create_data_set_csv(output_path):
    columns = ['full_name', 'scene', 'actor', 'obj', 'action', 'direct', 'dis', 'intensity', 'video_fn']
    data_set = {}
    count = 1
    for scene in scenes:
        for actor in actors:
            for obj in objects:
                for action in actions:
                    for direction in directions:
                        for dis in diss:
                            for intensity in intensitys:
                                data_set[count] = ["_".join([actor, obj, action, direction, dis, intensity]),
                                                   scene, actor, obj, action, direction, dis, intensity, '']

                                count += 1

    df = pd.DataFrame.from_dict(data_set, orient='index', columns=columns)

    df.to_csv(output_path, sep='\t')

    return


def get_action_names():
    files = ['20220829014414.json', '20220829014832.json', '20220829014659.json', '20220829014536.json']

    act_names = []
    for file in files:
        data = LoadBasic.load_json("actions", fn=file)
        for ele in data:
            actionName = ele['ActionName']
            if actionName not in act_names:
                act_names.append(actionName)

    return act_names


def adjust_data(data):
    for frame in range(len(data)):
        data[frame]['Skeleton'].append(position)
        for skeleton_point in data[frame]['Skeleton']:
            skeleton_point['WorldPosition'] = skeleton_point['LocalPosition']
            skeleton_point['WorldRotation'] = skeleton_point['LocalRotation']
    return 0


def split_write_action_json(output_path):
    files = ['20220829014414.json', '20220829014832.json', '20220829014659.json', '20220829014536.json']

    act_names = []
    for file in files:
        data = LoadBasic.load_json("actions", fn=file)
        for ele in data:
            act_name = ele['ActionName']
            act_data = ele['ActionData']
            adjust_data(act_data)

            SaveBasic.save_json(data=act_data, path=output_path,
                                fn="{}.json".format(act_name))


if __name__ == '__main__':
    # actions = get_action_names()
    # output_path = 'data_set.csv'
    # create_data_set_csv(output_path)

    root_path = "../../GGCamera_data/action_raw"
    split_write_action_json(root_path)
