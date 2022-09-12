import os


def pre_process_single_action_data(action_data, action_name="default", intensity=1, distance=2):
    # 1. raw data
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
    data_root = "../GGCamera_data/"
