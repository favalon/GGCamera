def output_cameras_track(points, focus, frame_offset=1):
    output_list = []
    for i, point in enumerate(points):
        frame = i + frame_offset
        new_frame = {
                        "Frame": 1,
                        "FocusedName": "Fropp",
                        "LocalPosition": {
                            "x": -0.5922043,
                            "y": -0.0287083387,
                            "z": -0.107094161
                        },
                        "LocalRotation": {
                            "x": 0.0,
                            "y": 318.63266,
                            "z": 0.0
                        },
                        "ModelWorldPosition": {
                            "x": 0.515219033,
                            "y": 0.0287083387,
                            "z": -0.3110055
                        },
                        "ModelWorldRotation": {
                            "x": 0.0,
                            "y": 41.36733,
                            "z": 0.0
                        }
                    },

        output_list.append(new_frame)

    return output_list
