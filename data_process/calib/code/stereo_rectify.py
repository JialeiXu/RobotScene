import cv2
import argparse
from glob import glob
import os
import yaml
from scipy.spatial.transform import Rotation as scipy_R


import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',default='/nas/xujl/datasets/parse_data/')
    args = parser.parse_args()
    scene_list = sorted(glob(args.root_path+'*'))
    for scene in scene_list:
        left_list = sorted(glob(os.path.join(scene,'cam3_compressed')+'/*'))
        right_list = sorted(glob(os.path.join(scene,'cam0_compressed')+'/*'))
        assert len(left_list) == len(right_list)
        for i in range(right_list.__len__()):
            # img_l = cv2.imread(left_list[i])
            # img_r = cv2.imread(right_list[i])
            img_l = cv2.imread('../tmp/stereo/data/left.png')
            img_r = cv2.imread('../tmp/stereo/data/right.png')
            R = np.array([[0.9999882824306063,-0.004461502627507186,0.0018788283025750689],
                [0.004460880499137978,0.9999899940476066,0.00033518627660447333],
                [-0.0018803049375622965,-0.00032680112049981465,0.9999981788255263
                ]])
            T = np.array([0.3977850913432908,-0.007340435017946437,-0.003899879401531515])
            ex = np.eye(4)
            ex[:3,:3],ex[:3,3] = R,T
            ex = np.linalg.pinv(ex)
            R,T = ex[:3,:3],ex[:3,3]
            left_K = np.array(yaml.load(open('lidar_camera_calib/left_iv1/intrinsics.yaml'),Loader=yaml.Loader)['camera_matrix']['data']).reshape(3,3)
            righ_K = np.array(yaml.load(open('lidar_camera_calib/right_iv1/intrinsics.yaml'),Loader=yaml.Loader)['camera_matrix']['data']).reshape(3,3)
            iv2left = np.loadtxt('calib/left_iv1/extrinsics.txt')
            iv2right = np.loadtxt('calib/right_iv1/extrinsics.txt')
            l2r = iv2right @ np.linalg.pinv(iv2left)
            R = l2r[:3,:3]
            #T = l2r[:3, 3]

            #
            # print(scipy_R.from_matrix(l2r[:3,:3]).as_rotvec())
            # print(scipy_R.from_matrix(R).as_rotvec())
            #
            # print(T,l2r[:3,3])
            # H = righ_K @ ex[:3,:3] @ np.linalg.pinv(left_K)
            # H = righ_K @ l2r[:3,:3] @ np.linalg.pinv(left_K)
            dist_coefs = np.array([0, 0, 0., 0., 0])
            rgb_shape = (2880, 1860)
            w,h = rgb_shape
            #new_right = cv2.warpPerspective(img_l, H, (w, h)).astype(np.uint8)

            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
                cv2.stereoRectify(left_K, dist_coefs, righ_K, dist_coefs, rgb_shape, R, T,
                                  alpha=-1)
            mapL1, mapL2 = cv2.initUndistortRectifyMap(left_K, dist_coefs, R1, P1, rgb_shape,
                                                       cv2.CV_32FC1)  # cv2.CV_16SC2)
            mapR1, mapR2 = cv2.initUndistortRectifyMap(righ_K, dist_coefs, R2, P2, rgb_shape,
                                                       cv2.CV_32FC1)  # cv2.CV_16SC2)
            rectL = cv2.remap(img_l, mapL1, mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(img_r, mapR1, mapR2, cv2.INTER_LINEAR)
            add = cv2.add(rectL,rectR)
            #add = cv2.add(new_right,img_r)
            cv2.imwrite('tmp/stereo/add.jpg',add)
            cv2.imwrite('tmp/stereo/left.jpg',img_l)
            cv2.imwrite('tmp/stereo/left_rec.jpg',rectL)
            cv2.imwrite('tmp/stereo/right.jpg',img_r)
            cv2.imwrite('tmp/stereo/right_rec.jpg', rectR)
            #cv2.imwrite('tmp/stereo/right_new.jpg',new_right)
            E = np.eye(4)
            print(T)
            print(l2r[:3, 3])

            E[:3,:3] = R
            E[:3,3] = T
            np.savetxt('calib/left2right.txt',E)
            break
        break