def important_point_selection(peak, valley):

    peak_cursor = 0
    valley_cursor = 0
    node = []

    while peak_cursor < len(peak) or valley_cursor < len(valley):

        if valley_cursor == len(valley):
            node.append([peak[peak_cursor], 1])
            peak_cursor += 1
        elif peak_cursor == len(peak):
            node.append([valley[valley_cursor], 1])
            valley_cursor += 1
        else:
            if peak[peak_cursor] < valley[valley_cursor]:
                current_min_index = peak[peak_cursor]
                node.append([peak[peak_cursor], 1])
                peak_cursor += 1

            else:
                current_min_index = valley[valley_cursor]
                node.append([valley[valley_cursor], 0])
                valley_cursor += 1

    selected_node = [node[0][0]]
    pre_flag = node[0][1]
    for i in node:
        cur_flag = i[1]
        if cur_flag != pre_flag:
            selected_node.append(i[0])
            pre_flag = cur_flag

    return selected_node


def get_all_focus_imp(skeleton_sequence, frames_imp_index):
    all_focus_imp_value = []
    for frame in range(skeleton_sequence.shape[0]):
        if frame in frames_imp_index:
            all_focus_imp_value.append(skeleton_sequence[frame, 0, :, :].mean(axis=0))

    return all_focus_imp_value