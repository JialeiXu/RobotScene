import json
import math
import os
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
import utm

from rotation import Rotation, Transform, Translation
import argparse
from multiprocessing import Pool
from glob import glob
import datetime
def write(path, data):
    with open(path, "w") as f:
        for l in data:
            f.write(" ".join([str(i) for i in l]) + "\n")


class Calib(object):
    def __init__(self, src, result):
        self.src = src
        self.result = result

    def fetch_imus(self):
        # 获取imu数据
        Imu = namedtuple("imu", ["timestamp", "data"])
        imus = []
        for file in Path(self.src).rglob("*"):
            if file.name.startswith("."):
                continue
            if file.name.endswith(".pose"):
                imu_data = self.read(str(file))
                imus.extend([Imu(float(data["timestamp"]), data) for data in imu_data])
        imus = sorted(imus, key=lambda x: x.timestamp)
        return imus

    def read(self, filepath):
        with open(filepath) as f:
            data = json.loads(f.read())
        return data

    def make_cz_rt(self, pose):
        # import pdb;pdb.set_trace()
        roll = math.radians(pose.data["euler"][0])
        pitch = math.radians(-pose.data["euler"][1])
        yaw = math.radians(-pose.data["euler"][2])
        # import pdb;pdb.set_trace()
        utm_coor = utm.from_latlon(pose.data["position"][0], pose.data["position"][1])
        tx, ty, tz = utm_coor[1], -utm_coor[0], pose.data["position"][2]
        r = Rotation.from_eulerangle([roll, pitch, yaw], "zyx").rotation
        t = Translation.from_list([tx, ty, tz]).translation
        return Transform.make_extrinsics(r, t)

    def loads_ins_pose(self, pcd_imus):
        T0 = None
        ins_poses = []
        for index, info in enumerate(pcd_imus):
            rt = self.make_cz_rt(info)
            ins_poses.append(rt)
        return ins_poses

    def evo_tum_export(self, pcd_imus):
        # lidar_poses = self.loads_lidar_pose(self.lidar_pose_path)
        ins_poses = self.loads_ins_pose(pcd_imus)
        I0 = ins_poses[0]
        ins = []
        res = []
        i = 0
        for ins_rt, imu in zip(ins_poses[::1], pcd_imus):
            # import pdb;pdb.set_trace()
            # 绝对位置
            q_ins = Rotation(ins_rt[:3, :3]).to_quaternion()
            t_ins = ins_rt[:3, 3]
            # 相对位置
            re_ins_rt = np.linalg.inv(I0)@ins_rt
            re_q_ins = Rotation(re_ins_rt[:3, :3]).to_quaternion()
            re_t_ins = re_ins_rt[:3, 3]
            # tum_lidar = [i, *t_lidar, *q_lidar[1:], q_lidar[0]]
            # tum_calib_ins = [i, *t_ins_calib, *q_ins_calib[1:], q_ins_calib[0]]
            tum_ins = [i, *t_ins, *q_ins[1:], q_ins[0]]
            res_ins = [imu.timestamp, *re_t_ins, *re_q_ins[1:], q_ins[0]]
            ins.append(tum_ins)
            res.append(res_ins)
            i += 1
        ins_file = os.path.join(os.path.dirname(self.result), "ins.txt")
        write(ins_file, ins)
        write(self.result, res)

    def run(self):
        imus = self.fetch_imus()
        self.evo_tum_export(imus)

def init(args):
    if os.path.exists(args.log):
        os.remove(args.log)

    current_time = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time())
    with open(args.log,'a+') as f:
        f.write(current_time)
def main_work(rank,args):
    #print(args.data_list)

    for i in range(rank,len(args.data_list),args.processes_num):
        data_file = args.data_list[i]
        print(data_file)
        #root = "output"
        #i = "2024-04-12-18-22-58"
        #pcd_imus = "./output/2024-04-12-18-22-58/pose/"
        pcd_imus = os.path.join(data_file,'pose/')
        #print(pcd_imus)
        output_path = os.path.join(args.output_path,data_file.split('/')[-1],data_file.split('/')[-1]+'.txt')
        os.makedirs(os.path.dirname(output_path),exist_ok=True)
        #result = os.path.join(args.output_path +, i, i + ".txt")
        calib = Calib(pcd_imus, output_path)
        calib.run()
        print("sucessfully!!!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes_num',default='100', type=int)
    parser.add_argument('--distribute',default=True, type=bool)
    parser.add_argument('--ParseData_path_list',default=[
                    #'/data/parse_data/2024-04-12',
                    #'/data/parse_data/2024-04-15',
                    #'/data/parse_data/2024-04-16',
                    '/data/parse_data/2024-04-18',
                                                      ],type=list)
    parser.add_argument('--output_path',default='/data/data/',type=str)
    parser.add_argument('--log',default='parse_pose_log.txt')
    args = parser.parse_args()

    init(args)

    data_list = []
    for i in args.ParseData_path_list:
        data_list.extend(sorted(glob(i + '*')))
    args.data_list = data_list

    if args.distribute:
        processes = Pool(args.processes_num)
        for rank in range(args.processes_num):
            processes.apply_async(main_work,args=(rank,args))
        processes.close()
        processes.join()
        print('dist done')
    else:
        main_work(0,args)
        print('done')

