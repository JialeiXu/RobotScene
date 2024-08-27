import configparser
import csv
import json
import os
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np

from camera import Camera
from pc import PointCloud
from rotation import Transform
from motion_compensa import read_11_pose, find_nearest_pose, transform
import pdb
from PIL import Image
from utils import gray_to_colormap
import argparse
from multiprocessing import Pool
from glob import glob
def make_xyzit_point_cloud(xyzitpf):
    """ Make a pointcloud object from xyz intensity timestamp point_index array.
        """
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.uint32), ('timestamp', np.float64)])
    xyzitpf = np.rec.fromarrays([xyzitpf[:, 0], xyzitpf[:, 1], xyzitpf[:, 2], xyzitpf[:, 3], xyzitpf[:, 4]], dtype=dt)
    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'intensity','timestamp'],
          'count': [1, 1, 1, 1, 1],
          'width': len(xyzitpf),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyzitpf),
          'type': ['F', 'F', 'F', 'U', 'F'],
          'size': [4, 4, 4, 4, 8],
          'data': 'binary'}
    pc = PointCloud(md, xyzitpf)
    return pc


def write_csv(dst_file: str, data: list, header: Union[list, str] = "", mod=1):
    encode = "utf-8"
    if mod == 2:
        encode = "utf-8-sig"
    if mod == 3:
        encode = "gbk"
    if isinstance(header, str):
        with open(dst_file, "w", encoding=encode) as f:
            if header and isinstance(header, str):
                header = header if header.endswith("\n") else header + "\n"
                f.write(header)
            for line in data:
                line = line if line.endswith("\n") else line + "\n"
                f.write(line)
    if isinstance(header, list):
        with open(dst_file, "w", encoding=encode) as f:
            writer = csv.DictWriter(f, header)
            writer.writeheader()
            writer.writerows(data)


# set the color by distance to the cloud
def getColor(cur_depth):
    cur_depth = abs(cur_depth)
    result_r = 0
    result_g = 0
    result_b = 0
    max_depth = 80
    min_depth = 5
    scale = (max_depth - min_depth) / 20
    # import pdb;pdb.set_trace()
    if cur_depth < min_depth:
        result_r = 0
        result_g = 0
        result_b = 255
    elif cur_depth < min_depth + scale:
        result_r = 0
        result_g = int((cur_depth - min_depth) / scale * 255)
        result_b = 255
    elif cur_depth < min_depth + scale * 2:
        result_r = 0
        result_g = 255
        result_b = 255 - int((cur_depth - min_depth - scale) / scale * 255)
    elif cur_depth < min_depth + scale * 4:
        result_r = int((cur_depth - min_depth - scale * 2) / scale * 255)
        result_g = 255
        result_b = 0
    elif cur_depth < min_depth + scale * 7:
        result_r = 255
        result_g = 255 - int((cur_depth - min_depth - scale * 4) / scale * 255)
        result_b = 0

    elif cur_depth < min_depth + scale * 10:
        result_r = 255
        result_g = 0
        result_b = int((cur_depth - min_depth - scale * 7) / scale * 255)
    else:
        result_r = 255
        result_g = 0
        result_b = 255
    return (result_r, result_g, result_b)


class Project:
    def __init__(self, img_path, camera_type, yaml_file, rt_file) -> None:
        self.extrinsics = np.loadtxt(rt_file)
        self.camera = Camera(img_path, camera_type, yaml_file)

    def make_color_map(self, rgb_points):
        # 过滤掉相机后面的点
        points = rgb_points[rgb_points[:, 2] > 0]
        total_color = list(map(getColor, points[:, 2]))
        # total_color = plt.cm.jet(abs(np_x))
        depth = points[:,2]
        return total_color, depth

    def project(self, objpoints, undistort_src_img):
        # objpoints = self.read_pcd(src_lidar_file)
        rgb_points = Transform.points_3Dto3D(objpoints, self.extrinsics)
        rgb_points = rgb_points[rgb_points[:, 2] > 0]
        proj_poxiel = Transform.points_3Dto2D(rgb_points, self.camera.camMtrx)

        # make poxiel color map
        # project 3D to 2D and draw in img

        #total_color, depth = self.make_color_map(rgb_points)
        depth = rgb_points[:,2]
        filter = (
            (proj_poxiel[:, 0] < self.camera.h)
            & (proj_poxiel[:, 0] >= 0)
            & (proj_poxiel[:, 1] < self.camera.w)
            & (proj_poxiel[:, 1] >= 0)
            & (rgb_points[:, 2] > 0)
        )
        proj_poxiel = proj_poxiel[np.where(filter)[0]]
        #colors = np.array(total_color)[np.where(filter)[0]]
        depth = np.where(depth > 250, np.ones_like(depth) * 250, depth)
        depth = np.array(depth * 256)[np.where(filter)[0]].astype(np.uint16)
        depth_map = np.zeros_like(undistort_src_img[:,:,0]).astype(np.uint16)

        # black_im = np.zeros(shape=undistort_src_img.shape)
        for point, depth_i in zip(proj_poxiel, depth):
            depth_map[int(point[1])][int(point[0])] = depth_i
        # img = cv.addWeighted(undistort_src_img, 0.7, black_im, 2, 0)
        # remove points that not in img
        return depth_map
    def __call__(self, src_lidar_file, is_undistorted, save_img_path,depth_map_path,depth_vis_path,pil_save=False):
        if is_undistorted:
            undistort_src_img = self.camera.image
        else:
            undistort_src_img = self.camera.undistort()
        # transform lidar points to camera coordinate and projected to img
        # undistort_img_path = str(Path(root).joinpath('undistort.png'))
        # project_img_path = str(Path(root).joinpath('project_img.png'))
        # cv.imwrite(undistort_img_path, undistort_src_img)

        #save undistort image
        if pil_save:
            Image.fromarray(undistort_src_img[:,:,0]).save(save_img_path)
        else:
            cv.imwrite(save_img_path,undistort_src_img)
        #save depth
        depth = self.project(src_lidar_file,undistort_src_img)
        Image.fromarray(depth).save(depth_map_path)

        #save vis_depth
        depth_color = gray_to_colormap(depth,max=80*200)
        depth_img_color = np.where(np.stack([depth,depth,depth],2)>0,depth_color,undistort_src_img)
        cv.imwrite(depth_vis_path,depth_img_color)
        #self.project(src_lidar_file, undistort_src_img)
        #cv.imwrite(project_img_path, undistort_src_img)


def parse_args(config_file):
    # 生成ConfigParser对象
    config = configparser.ConfigParser()
    # 读取配置文件
    config.read(config_file, encoding="utf-8")
    # common
    root = config.get("Common", "root")
    camera_type = config.get("Common", "camera_type")
    only_projection = config.getboolean("Common", "only_projection")
    src_points = config.get("Common", "src_points")
    is_undistorted = config.getboolean("Common", "is_undistorted")
    return root, camera_type, src_points, is_undistorted, only_projection


def find_nearest(array, data):
    # 二分法查找时间最近的pcd
    start, end = 0, len(array) - 1
    while start < end:
        mid_index = (start + end) // 2
        if array[mid_index].timestamp < data:
            start = mid_index + 1
        else:
            end = mid_index
    if abs(array[start].timestamp - data) > abs(array[start - 1].timestamp - data):
        return array[start - 1]
    else:
        return array[start]


def save_pcd(points, pcd_file):
    _num, _ = points.shape
    md = {
        "version": 0.7,
        "fields": ["x", "y", "z"],
        "size": [4, 4, 4],
        "type": ["F", "F", "F"],
        "count": [1, 1, 1],
        "width": _num,
        "height": 1,
        "viewpoint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "points": _num,
        "data": "binary",
    }
    dtypes = [("x", np.float32), ("y", np.float32), ("z", np.float32)]
    pc_data = np.zeros((_num), dtypes)
    pc_data["x"] = points[:, 0]
    pc_data["y"] = points[:, 1]
    pc_data["z"] = points[:, 2]

    p = PointCloud(md, pc_data)
    p.save_pcd(pcd_file)


def motion_compensa(pc, cam_timestamp, pose_timestamps, poses, RT_lidar_to_imu):
    """
    motion compensa
    """
    # 查找距离雷达每个点时间戳最近的ins pose和相机时间戳最近的pose
    #pdb.set_trace()
    inx_lidars, inx_cam = find_nearest_pose(pc, cam_timestamp, pose_timestamps)
    # 将雷达每一个点转换到对应相机那一刻
    points = transform(pc, inx_lidars, inx_cam, poses, RT_lidar_to_imu)
    pc.pc_data["x"] = points[:, 0]
    pc.pc_data["y"] = points[:, 1]
    pc.pc_data["z"] = points[:, 2]
    return points


class Item:
    def __init__(self, path) -> None:
        self.path = path

    @property
    def timestamp(self):
        return float(self.path.stem)


def run(rank,root, dst_path,args):
    ir_intrinsics = "lidar_camera_calib/ir/intrinsics.yaml"
    ir_extrinsics = "lidar_camera_calib/ir/extrinsics.txt"
    left_extrinsics = "lidar_camera_calib/left_iv1/extrinsics.txt"
    left_intrinsics = "lidar_camera_calib/left_iv1/intrinsics.yaml"
    right_extrinsics = "lidar_camera_calib/right_iv1/extrinsics.txt"
    right_intrinsics = "lidar_camera_calib/right_iv1/intrinsics.yaml"
    lidar_to_imu = "lidar_camera_calib/lidar_to_imu.txt"
    tdt2_to_rs = "lidar_camera_calib/iv2_to_rslidar.txt"
    tdt1_to_rs = "lidar_camera_calib/iv_to_rslidar.txt"

    tdt1_path = f"{root}/iv_points"
    tdt2_path = f"{root}/iv_points2"
    rs_path = f"{root}/rslidar_points"
    left_path = f"{root}/cam3"
    right_path = f"{root}/cam0"
    ir_path = f"{root}/ir"
    pose_path = f"{root}/pose"

    rt_tdt2_rs = np.loadtxt(tdt2_to_rs)
    rt_tdt1_rs = np.loadtxt(tdt1_to_rs)

    tdt1_pcds = sorted(
        [Item(pcd) for pcd in Path(tdt1_path).rglob("*.pcd")], key=lambda x: x.timestamp
    )
    tdt2_pcds = sorted(
        [Item(pcd) for pcd in Path(tdt2_path).rglob("*.pcd")], key=lambda x: x.timestamp
    )
    rs_pcds = sorted(
        [Item(pcd) for pcd in Path(rs_path).rglob("*.pcd")], key=lambda x: x.timestamp
    )
    #print('left_pngs',left_path)
    left_pngs = sorted(
        [Item(png) for png in Path(left_path).rglob("*.jpg")], key=lambda x: x.timestamp
    )
    right_pngs = sorted(
        [Item(png) for png in Path(right_path).rglob("*.jpg")],
        key=lambda x: x.timestamp,
    )
    ir_pngs = sorted(
        [Item(png) for png in Path(ir_path).rglob("*.png")], key=lambda x: x.timestamp
    )
    # 获取INS pose和时间戳
    pose_timestamps, poses = read_11_pose(pose_path)
    RT_lidar_to_imu = np.loadtxt(lidar_to_imu)

    #record_list = []
    #for inx, tdt1_pcd in enumerate(tdt1_pcds):
    for inx in range(rank,len(tdt1_pcds),args.processes_num):
        print('inx',inx)
        tdt1_pcd = tdt1_pcds[inx]
        if inx == 0:
            continue
        # if tdt1_pcd.path.name != "1713432583.699699000.pcd":
        #     continue
        #pdb.set_trace()
        tdt2_pcd = find_nearest(tdt2_pcds, tdt1_pcd.timestamp)
        rs_pcd = find_nearest(rs_pcds, tdt1_pcd.timestamp)
        left_png = find_nearest(left_pngs, tdt1_pcd.timestamp) #hear
        right_png = find_nearest(right_pngs, tdt1_pcd.timestamp)
        ir_png = find_nearest(ir_pngs, tdt1_pcd.timestamp)
        #pdb.set_trace()
        dst_all_points_pcd_path = Path(dst_path).joinpath(
            "all_points", tdt1_pcd.path.name.replace(".png", ".pcd")
        )
        dst_all_points_pcd_path.parent.mkdir(exist_ok=True, parents=True)

        dst_motion_points_pcd_path = Path(dst_path).joinpath(
            "motion_points", tdt1_pcd.path.name.replace(".png", ".pcd")
        )
        dst_motion_points_pcd_path.parent.mkdir(exist_ok=True, parents=True)

        # record_list = [
        #     "{},{},{},{},{},{}".format(
        #         tdt1_pcd.timestamp,
        #         tdt2_pcd.timestamp,
        #         rs_pcd.timestamp,
        #         left_png.timestamp,
        #         right_png.timestamp,
        #         ir_png.timestamp,
        #     )
        # ]
        record_list = [
                 "{},{},{},{},{},{}".format(
                     tdt1_pcd.path.name,
                     tdt2_pcd.path.name,
                     rs_pcd.path.name,
                     left_png.path.name,
                     right_png.path.name,
                     ir_png.path.name
                 )
        ]

        # combine
        pc_tdt2 = PointCloud.from_path(tdt2_pcd.path)
        # import pdb;pdb.set_trace()

        pc_rs = PointCloud.from_path(rs_pcd.path)
        pc_tdt1 = PointCloud.from_path(tdt1_pcd.path)
        trans_tdt2_points = Transform.points_3Dto3D(pc_tdt2.points_np,  np.linalg.inv(rt_tdt1_rs) @ rt_tdt2_rs)
        trans_rs_points = Transform.points_3Dto3D(pc_rs.points_np, np.linalg.inv(rt_tdt1_rs))
        # trans_tdt1_points = Transform.points_3Dto3D(pc_tdt1.points_np, rt_tdt1_rs)
        all_tdt2_points = np.hstack((trans_tdt2_points, pc_tdt2.data_np[:, 3:5]))
        all_tdt1_points = np.hstack((pc_tdt1.points_np, pc_tdt1.data_np[:, 3:5]))
        all_rs_points = np.hstack((trans_rs_points, 
                                   pc_rs.pc_data['intensity'].reshape(-1, 1), 
                                   pc_rs.pc_data['timestamp'].reshape(-1, 1)))
        # import pdb;pdb.set_trace()
        total_src_points = np.vstack(
            (all_tdt2_points, all_tdt1_points, all_rs_points)
        )
        # import pdb;pdb.set_trace()
        new_pc = make_xyzit_point_cloud(total_src_points)
        if args.save_pcd:
            new_pc.save_pcd(dst_all_points_pcd_path)
        #pdb.set_trace()
        # 雷达运动补偿
        total_points = motion_compensa(new_pc, left_png.timestamp, pose_timestamps, poses, RT_lidar_to_imu)
        if args.save_pcd:
            new_pc.save_pcd(dst_motion_points_pcd_path)

        # 投影左相机验证
        project = Project(
            str(left_png.path), "pinhole", left_intrinsics, left_extrinsics
        )
        dst_left_path = Path(dst_path).joinpath("left", left_png.path.name)
        dst_left_path.parent.mkdir(exist_ok=True, parents=True)
        dst_left_depth_path = Path(dst_path).joinpath("left_depth", left_png.path.name)
        dst_left_depth_path.parent.mkdir(exist_ok=True, parents=True)
        dst_left_depth_vis_path = Path(dst_path).joinpath("left_depth_vis", left_png.path.name)
        dst_left_depth_vis_path.parent.mkdir(exist_ok=True, parents=True)
        #pdb.set_trace()
        project(total_points, False, str(dst_left_path),str(dst_left_depth_path)[:-3]+'png',str(dst_left_depth_vis_path))

        # 投影右相机验证
        project = Project(
            str(right_png.path), "pinhole", right_intrinsics, right_extrinsics
        )
        dst_right_path = Path(dst_path).joinpath("right", right_png.path.name)
        dst_right_path.parent.mkdir(exist_ok=True, parents=True)
        dst_right_depth_path = Path(dst_path).joinpath("right_deth", right_png.path.name)
        dst_right_depth_path.parent.mkdir(exist_ok=True, parents=True)
        dst_right_depth_vis_path = Path(dst_path).joinpath("right_depth_vis", right_png.path.name)
        dst_right_depth_vis_path.parent.mkdir(exist_ok=True, parents=True)
        project(total_points, False, str(dst_right_path),str(dst_right_depth_path)[:-3]+'png',str(dst_right_depth_vis_path))

        # 投影ir验证
        project = Project(str(ir_png.path), "pinhole", ir_intrinsics, ir_extrinsics)
        dst_ir_path = Path(dst_path).joinpath("ir", ir_png.path.name)
        dst_ir_path.parent.mkdir(exist_ok=True, parents=True)
        dst_ir_path_depth = Path(dst_path).joinpath("ir_depth", ir_png.path.name)
        dst_ir_path_depth.parent.mkdir(exist_ok=True, parents=True)
        dst_ir_path_depth_vis = Path(dst_path).joinpath("ir_depth_vis", ir_png.path.name)
        dst_ir_path_depth_vis.parent.mkdir(exist_ok=True, parents=True)
        project(total_points, False, str(dst_ir_path),str(dst_ir_path_depth),str(dst_ir_path_depth_vis),pil_save=True)

        #timestamp log
        os.makedirs(os.path.join(dst_path,'timestamp_log'),exist_ok=True)
        with open(os.path.join(dst_path,'timestamp_log',str(tdt1_pcd.path.name)[:-4] + '.json'),'w') as json_f:
            json.dump(record_list,json_f)
        #write_csv(dst_file=os.path.join(dst_path,'timestamp_log',str(tdt1_pcd.timestamp) + '.csv'),data=record_list)

    #write_csv(dst_file=os.path.join(dst_path, "record.csv"), data=record_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes_num',default=96,type=int)
    parser.add_argument('--distributed',default=True,type=bool)
    parser.add_argument('--parse_data_path',type=list,default=[
                    '/data_nfs/datasets/parse_data/2024-04-18-15-37-36',
                    '/data_nfs/datasets/parse_data/2024-04-18-15-54-35',
                    '/data_nfs/datasets/parse_data/2024-04-18-17-29-30',
                    '/data_nfs/datasets/parse_data/2024-04-18-20-03-50',
                    '/data_nfs/datasets/parse_data/2024-04-19-16-29-27',
                    '/data_nfs/datasets/parse_data/2024-04-19-19-29-44',

                    ])
    parser.add_argument('--output_path',default='/data_nfs/datasets/data/')
    parser.add_argument('--save_pcd',default=False)
    args = parser.parse_args()
    parse_data_list = []
    for path in args.parse_data_path:
        parse_data_list.extend(sorted(glob(path + '*')))
    args.parse_data_list = parse_data_list
    print('args.parse_data_list:',args.parse_data_list)
    for parse_data_path in args.parse_data_list:
        if args.distributed:
            processes = Pool(args.processes_num)
            for rank in range(args.processes_num):
                processes.apply_async(run, args=(rank,parse_data_path,args.output_path + parse_data_path.split('/')[-1], args))
            processes.close()
            processes.join()
            print('dist done')
        else:
            args.processes_num = 1
            #main_work(0, args)
            run(0,parse_data_path, args.output_path + parse_data_path.split('/')[-1], args)
            print('done')


    #main()
