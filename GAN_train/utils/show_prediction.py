import math
import os

import cv2
import numpy as np

# basic info about limb composition
joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]]
# for plot usage
mm_colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]


def draw_pose_figure(coors, height=360, width=640, limb_thickness=4):
    canvas = np.ones([height, width, 3]) * 255
    canvas = canvas.astype(np.uint8)
    limb_type = 0
    for joint_relation in joint_to_limb_heatmap_relationship:
        if (limb_type >= 17):
            break
        joint_coords = coors[joint_relation]
        for joint in joint_coords:  # Draw circles at every joint
            '''
            haoran added print
            '''
            # print('joint',joint)
            cv2.circle(canvas, tuple(joint[0:2].astype(
                int)), 4, (0, 0, 0), thickness=-1)
        coords_center = tuple(
            np.round(np.mean(joint_coords, 0)).astype(int))
        limb_dir = joint_coords[0, :] - joint_coords[1, :]
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv2.ellipse2Poly(
            coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, mm_colors[limb_type])
        limb_type += 1
    return canvas


def vis_npis(poses, outdir, aud=None):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    neglect = [14, 15, 16, 17]

    for t in range(poses.shape[0]):
        # break
        canvas = np.ones((256, 500, 3), np.uint8) * 255

        thisPeak = poses[t]
        for i in range(18):
            if i in neglect:
                continue
            if thisPeak[i, 0] == -1:
                continue
            cv2.circle(canvas, tuple(thisPeak[i, 0:2].astype(int)), 4, colors[i], thickness=-1)

        for i in range(17):
            limbid = np.array(limbSeq[i]) - 1
            if limbid[0] in neglect or limbid[1] in neglect:
                continue
            X = thisPeak[[limbid[0], limbid[1]], 1]
            Y = thisPeak[[limbid[0], limbid[1]], 0]
            if X[0] == -1 or Y[0] == -1 or X[1] == -1 or Y[1] == -1:
                continue
            stickwidth = 4
            cur_canvas = canvas.copy()
            mX = np.mean(X)
            mY = np.mean(Y)
            # print(X, Y, limbid)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            # print(i, n, int(mY), int(mX), limbid, X, Y)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        if aud is not None:
            if aud[:, t] == 1:
                cv2.circle(canvas, (30, 30), 20, (0, 0, 255), -1)
                # canvas = cv2.copyMakeBorder(canvas,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
        cv2.imwrite(os.path.join(outdir, 'frame{0:03d}.png'.format(t)), canvas)


def save_my_batch_images(fake_coors, target_folder, isDebug=False):
    fake_reshape_coors = fake_coors.reshape([-1, 18, 2])
    for i in range(fake_reshape_coors.shape[0]):
        idx = str("%03d" % i)
        output_dir = target_folder + '/' + idx + '.jpeg'
        fake_img = draw_pose_figure(fake_reshape_coors[i])

        if isDebug == True:
            cv2.imshow('canvas', fake_img)
            cv2.waitKey(500)
        else:
            cv2.imwrite(output_dir, fake_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
