import glob
import os.path

import cv2


def process_single_video(video_fp, image_path, kps=1, extend='.png'):
    cap = cv2.VideoCapture(video_fp)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print("processing video {}".format(video_fp.split("\\")[-1]))
    action_name = video_fp.split("\\")[-1].split('.')[0]
    # exit()
    hop = round(fps / kps)
    curr_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if curr_frame % hop == 0:
            name = image_path + action_name + "_" + str(curr_frame) + extend
            cv2.imwrite(name, frame)
        curr_frame += 1
        cap.release()


def grab_video_fp(root):
    all_video_fp = glob.glob(os.path.join(root, "*/video/*.mp4"))
    return all_video_fp


def get_image_path(video_fp):
    fp_token = video_fp.split("\\")
    image_path_token = fp_token[:-2]
    image_path = "\\".join(image_path_token) + "\\images\\"

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    return image_path


def main():
    # data root
    root_fp = "C:\\Users\\TRA\\Desktop\\dataset"
    all_video_fp = grab_video_fp(root_fp)
    for video_fp in all_video_fp:
        image_path = get_image_path(video_fp)
        process_single_video(video_fp, image_path)
        # print(all_video_fp)
    return


if __name__ == '__main__':
    main()
