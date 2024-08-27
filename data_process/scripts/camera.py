import cv2 as cv
import numpy as np
import yaml


class Camera:
    def __init__(self, image_path, type, yaml_file) -> None:
        self.image_path = image_path
        self.camera_type = type
        self.camMtrx, self.dis_coeff = self.get_intrinsics(yaml_file)
        self.w, self.h = self.image.shape[0], self.image.shape[1]

    @property
    def image(self):
        return cv.imread(self.image_path)

    def get_intrinsics(self, yaml_file):
        with open(yaml_file) as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        camera_mat, dis_coeff = np.array(data['camera_matrix']['data']).reshape(3, 3), np.array(data['distortion_coefficients']['data'])
        return camera_mat, dis_coeff

    def get_cv_intrinsics(self, yaml_file):
        # 打开yaml文件
        f = cv.FileStorage(str(yaml_file), cv.FileStorage_READ)
        camera_mat = f.getNode('camera_matrix').mat().reshape(3, 3)
        dist_coeff = f.getNode('dist_coeffs').mat()
        f.release()
        return camera_mat, dist_coeff

    def _pinhole_undistort(self):
        # 针孔相机去畸变
        dst = cv.undistort(self.image, self.camMtrx, self.dis_coeff, None, None)
        # 剪裁图像
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        return dst

    def _fisheye_undistort(self):
        # 鱼眼去畸变
        Knew = self.camMtrx.copy()
        # Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]
        undistorted_img = cv.fisheye.undistortImage(self.image, self.camMtrx, self.dis_coeff, Knew=Knew)
        return undistorted_img
    
    def undistort(self):
        if self.camera_type == 'fisheye':
            return self._fisheye_undistort()
        elif self.camera_type == 'pinhole':
            return self._pinhole_undistort()
        else:
            raise TypeError("Error camera type")
