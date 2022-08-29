def output_cameras_track(points, focus, angles, frame_offset=1, focus_side='left'):
    output_list = []
    for i, point in enumerate(points):
        frame = i + frame_offset

        new_frame = {
            "Frame": frame,
            "FocusedName": "Fropp",
            "LocalPosition": {
                "x": points[i][0],  # + focus[i][0],
                "y": points[i][2] if points[i][2] > 0.1 else 0.1,
                "z": points[i][1]  # + focus[i][1]
            },
            "LocalRotation": {
                "x": angles[i][2] - 90,
                "y": - angles[i][1] if 'r' in focus_side else angles[i][1],
                "z": 0
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

        output_list.append(new_frame)

    return {'data': output_list}
