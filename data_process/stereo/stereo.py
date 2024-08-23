import cv2
import numpy as np
from PIL import Image
import json
import yaml
from scipy.spatial.transform import Rotation as scipy_R

def rotationMatrixToEulerAngles(R):
    r = scipy_R.from_matrix(R)
    res = r.as_rotvec()
    return res #for ceres




left = np.array(Image.open('./tmp/stereo/data/left.png'))
right = np.array(Image.open('./tmp/stereo/data/right.png'))
#json_f = json.load(open('/data3/xujl/project/CETC/dataset/CETC/data/2017-01-01-08-16-16_addT/record.json','r'))


left_extrinsics = "./scripts/lidar_camera_calib/left_iv1/extrinsics.txt"
left_intrinsics = "./scripts/lidar_camera_calib/left_iv1/intrinsics.yaml"
right_extrinsics = "./scripts/lidar_camera_calib/right_iv1/extrinsics.txt"
right_intrinsics = "./scripts/lidar_camera_calib/right_iv1/intrinsics.yaml"

left_k = np.array(yaml.load(open(left_intrinsics),Loader=yaml.Loader)['camera_matrix']['data']).reshape(3,3)
right_k = np.array(yaml.load(open(right_intrinsics),Loader=yaml.Loader)['camera_matrix']['data']).reshape(3,3)
E_left = np.loadtxt(left_extrinsics)
E_right = np.loadtxt(right_extrinsics)


ex = ex = E_left @ np.linalg.pinv(E_right)
print(left_k,right_k)
print('old ex:',rotationMatrixToEulerAngles(ex[:3,:3]))

left_k = np.array([
		[
			2648.0604074911555,
			0,
			1446.7696817161189
		],
		[
			0,
			2650.001837389897,
			959.8814806976726
		],
		[
			0,
			0,
			1
		]]).reshape(3,3)
right_k = np.array([
		[
			2641.402961835155,
			0,
			1438.437767022184
		],
		[
			0,
			2643.975759684023,
			943.3692635556153
		],
		[
			0,
			0,
			1
		]
	]).reshape(3,3)
print(left_k,right_k)
ex = np.array([
		[
			0.9999882824306063,
			-0.004461502627507186,
			0.0018788283025750689
		],
		[
			0.004460880499137978,
			0.9999899940476066,
			0.00033518627660447333
		],
		[
			-0.0018803049375622965,
			-0.00032680112049981465,
			0.9999981788255263
		]
	]).reshape(3,3)
print('new ex:',rotationMatrixToEulerAngles(ex[:3,:3]))
#H = left_k @ E_left[:3,:3] @ np.linalg.pinv( E_right[:3,:3]) @ np.linalg.pinv(right_k)
H = left_k  @ ex @ np.linalg.pinv(right_k)

h,w,_ = left.shape
right = right.astype(np.float32)

new_rgb = cv2.warpPerspective(right,H,(w,h)).astype(np.uint8)
Image.fromarray(new_rgb).save('./tmp/stereo/data/new_rgb.png')

add_img = cv2.add(new_rgb,left)
cv2.imwrite('./tmp/stereo/data/add.png',add_img)
print('done')