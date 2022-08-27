import pandas as pd

scenes = ['castle', 'forrest', 'side']
actors = ['man']
objects = ['man', 'women']
actions = ['stand_front_push_4', 'run_front_jump_1', 'stand_front_jump_5', 'interact_pickUp_04', 'stand_front_climb_1',
            'stand_front_swing_4', 'stand_front_interact_2', 'kneel_1_prostrate_1', 'jump_left_fall_1',
            'stand_front_pull_5', 'stand_front_kick_4', 'stand_front_punch_4', 'stand_front_clap_4']
diss = ['2', '4', '8']
directions = ['0', '90', '180', '270']
intensitys = ['0.2', '0.6', '1.0']


def create_data_set_csv(output_path):
    columns = ['full_name', 'scene', 'actor', 'obj', 'action', 'direct', 'dis',  'intensity', 'video_fn']
    data_set = {}
    count = 1
    for scene in scenes:
        for actor in actors:
            for obj in objects:
                for action in actions:
                    for direction in directions:
                        for dis in diss:
                            for intensity in intensitys:
                                data_set[count] = ["_".join([scene, actor, obj,action,  direction, dis, intensity]),
                                               scene, actor, obj, action, direction, dis, intensity, '']

                                count += 1

    df = pd.DataFrame.from_dict(data_set, orient='index', columns=columns)

    df.to_csv(output_path, sep='\t')

    return


if __name__ == '__main__':
    output_path = 'local_data/data_set1.csv'
    create_data_set_csv(output_path)