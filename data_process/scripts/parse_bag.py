import copy
import json
import logging
import os
import pdb

import cv2
import rosbag
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from glob import glob
import argparse
from multiprocessing import Pool
from multiprocessing import Process
from datetime import datetime

bridge = CvBridge()
import sys

log = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d %(message)s"
)
log.addHandler(handler)
log.setLevel(logging.INFO)


PCD_BINARY_TEMPLATE = """VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA binary
"""

PCD_BINARY_TEMPLATE_RS = """VERSION 0.7
FIELDS x y z intensity timestamp
SIZE 4 4 4 4 8
TYPE F F F U F
COUNT 1 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA binary
"""
PCD_BINARY_TEMPLATE_TDT = """VERSION 0.7
FIELDS x y z intensity timestamp scan_id scan_idx
SIZE 4 4 4 4 8 2 2
TYPE F F F U F U U
COUNT 1 1 1 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA binary
"""


def to_pcd_binary_rs(pcdpath, msg):
    f = open(pcdpath, "wb")
    lidar = list(pc2.read_points(msg))
    # print(lidar[0])
    header = copy.deepcopy(PCD_BINARY_TEMPLATE_RS).format(len(lidar), len(lidar))
    f.write(header.encode())
    import struct

    for pi in lidar:
        #print('pi',pi)
        h = struct.pack("<fffId", pi[0], pi[1], pi[2], int(pi[3]), pi[5]) #need to be edit

        # h = struct.pack("<fffI", pi[0], pi[1], pi[2], int(pi[3]))
        f.write(h)
    f.close()


def to_pcd_binary_tdt(pcdpath, msg):
    f = open(pcdpath, "wb")
    lidar = list(pc2.read_points(msg))
    # print(lidar[0])
    header = copy.deepcopy(PCD_BINARY_TEMPLATE_TDT).format(len(lidar), len(lidar))
    f.write(header.encode())
    import struct

    for pi in lidar:
        h = struct.pack(
            "<fffIdHH", pi[0], pi[1], pi[2], int(pi[3]), pi[4], pi[5], pi[6]
        )
        f.write(h)
    f.close()


def to_img(msg, output, tsp="", compressed=False, depth=False):
    if depth:
        enc = "passthrough"
    else:
        enc = "bgr8"
    try:
        if compressed:
            cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=enc)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=enc)

        # if depth:
        #     im = cv_image
        # else:
        #     im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        im = cv_image

        if tsp:
            if compressed:
                cv2.imwrite("{}/{}.jpg".format(output, tsp), im,[int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite("{}/{}.png".format(output, tsp), im)
        return im
    except CvBridgeError as e:
        print(e)


def to_imu(dst, imus):
    for timestamp, imuinfo in imus.items():
        imupath = os.path.join(dst, "current_imu_" + str(timestamp) + ".imu")
        if os.path.exists(imupath):
            continue
        with open(imupath, "w") as f:
            f.write(json.dumps(imuinfo, indent=4, separators=(",", ":")))


def to_pose(dst, poses):
    for timestamp, poseinfo in poses.items():
        posepath = os.path.join(dst, "current_pose_" + str(timestamp) + ".pose")
        if os.path.exists(posepath):
            continue
        with open(posepath, "w") as f:
            f.write(json.dumps(poseinfo, indent=4, separators=(",", ":")))


def parse_bag(bag_file: str, topic, output) -> None:
    """Parse bag file.

    Args:
    ----
        bag_file: use rosbag record
    """
    os.makedirs("{}/cam0".format(output), exist_ok=True)
    os.makedirs("{}/cam3".format(output), exist_ok=True)
    os.makedirs("{}/iv_points".format(output), exist_ok=True)
    os.makedirs("{}/iv_points2".format(output), exist_ok=True)
    os.makedirs("{}/rslidar_points".format(output), exist_ok=True)
    os.makedirs("{}/ir".format(output), exist_ok=True)
    os.makedirs("{}/imu".format(output), exist_ok=True)
    os.makedirs("{}/pose".format(output), exist_ok=True)
    poses = {}
    imus = {}
    with rosbag.Bag(bag_file, "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=topic):

            # print(msg.header.stamp.to_sec())
            # timestr = "%.9f" % msg.header.stamp.to_sec()

            # if topic != "/gnss_pub_node/current_pose":
            #     continue
            # pdb.set_trace()
            timestr = '%.9f' % msg.header.stamp.to_sec()
            _timestr_list = timestr.split('.')
            _ns = _timestr_list[1].lstrip('000') + '000'
            cam_timestr = f'{_timestr_list[0]}.{_ns.zfill(9)}'

            # if topic == "/gnss_pub_node/current_pose":
            #     timestr = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs).zfill(9))
            # else:
            #     timestr = (
            #     str(msg.header.stamp.secs)
            #     + "."
            #     + str(msg.header.stamp.nsecs).zfill(6).ljust(9, "0")
            # )
            # 处理cam时间戳前置0的方法，已经弃用
            if topic == "/v4l2_camera/camera0/image/compressed":
                to_img(
                    msg,
                    "{}/cam0".format(output),
                    cam_timestr,
                    depth=False,
                    compressed=True,
                )
            elif topic == "/v4l2_camera/camera3/image/compressed":
                to_img(
                    msg,
                    "{}/cam3".format(output),
                    cam_timestr,
                    depth=False,
                    compressed=True,
                )
            elif topic == "/iv_points_new":
                to_pcd_binary_tdt("{}/iv_points/{}.pcd".format(output, timestr), msg)
            elif topic == "/iv_points2_new":
                to_pcd_binary_tdt("{}/iv_points2/{}.pcd".format(output, timestr), msg)
            if topic == "/rslidar_points":
                to_pcd_binary_rs(
                    "{}/rslidar_points/{}.pcd".format(output, timestr), msg
                )
            elif topic == "/ir_image":
                to_img(msg, "{}/ir".format(output), timestr, depth=True)
            elif topic == "/gnss_pub_node/current_pose":
                if not msg.header.stamp.secs in poses:
                    poses.update({msg.header.stamp.secs: []})
                poses[msg.header.stamp.secs].append(
                    {
                        "timestamp": timestr,
                        "position": [
                            msg.latitude,
                            msg.longitude,
                            msg.altitude,
                        ],  # 经度，纬度，海拔
                        # "orientation":[msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w], # 四元素(x,y,z,w)
                        # "coordinate":[msg.twist.linear.y,msg.twist.linear.z,msg.pose.position.z], # 纬度，经度，高度(单位 m)
                        "euler": [msg.roll, msg.pitch, msg.yaw],  # roll, pitch, yaw
                        "speed": [
                            msg.velocity_north,
                            msg.velocity_east,
                            msg.velocity_up,
                        ],  # north, east, up
                        "stars": [msg.NSV1, msg.NSV2],  # 前天线星数， 后天线星数
                        "status": msg.status,
                    }
                )
                # print("Extract pose {}".format(timestr))
            elif topic == "/gnss_pub_node/current_imu":
                if not msg.header.stamp.secs in imus:
                    imus.update({msg.header.stamp.secs: []})
                imus[msg.header.stamp.secs].append(
                    {
                        "timestamp": timestr,
                        "orientation": [
                            msg.orientation.x,
                            msg.orientation.y,
                            msg.orientation.z,
                            msg.orientation.w,
                        ],  # x,y,z,w
                        "angular_velocity": [
                            msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z,
                        ],  # 陀螺仪x轴角速度，陀螺仪y轴角速度(单位 度/秒)，陀螺仪z轴角速度
                        "linear_acceleration": [
                            msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z,
                        ],  # 加速度计x轴加速度(单位 g)，加速度计y轴加速度，加速度计z轴加速度
                    }
                )
            # print(timestr)
    to_imu("{}/imu".format(output), imus)
    to_pose("{}/pose".format(output), poses)


def main_work(rank, args):


    for i in range(rank,len(args.rosbag_topic_list),args.processes_num):
        bag_file,topic = args.rosbag_topic_list[i]
        output = os.path.join(args.output_path,bag_file.split('/')[-1].replace('.bag',''))
        print('i=',i,bag_file,topic,output)
        #try:
        parse_bag(bag_file,topic, output)
        with open(args.log, 'a+') as f:
            f.write(bag_file +' ' + topic + ' done' + '\n')
        # except:
        #     with open(args.log, 'a+') as f:
        #         f.write(bag_file +' ' + topic + ' error' + '\n')

def init(args):
    if os.path.exists(args.log):
        os.remove(args.log)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.log,'a+') as f:
        f.write('Timestamp: ' + current_time + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes_num',default=96, type=int)
    parser.add_argument('--distribute',default=True, type=bool)
    parser.add_argument('--rosbag_path_list',default=[
                    '/data_nvme/rosbag/20240426/',
                    #'/data_nfs/datasets/rosbag/20240518/2024-05-18-15-35-23',
                    #'/data_nfs/datasets/rosbag/20240531/2024-05-31-10-10-42',
                    #'/nvme/rosbag/ir-20240522/',
                                                      ],type=list)
    parser.add_argument('--output_path',default='/data_nfs/datasets/parse_data',type=str)
    parser.add_argument('--specific_rosbag',default='',type=str)
    parser.add_argument('--log',default='./log.txt',type=str)
    parser.add_argument('--topic_list',default=[
        "/v4l2_camera/camera0/image/compressed",
        "/v4l2_camera/camera3/image/compressed",
        "/iv_points_new",
        "/iv_points2_new",
        "/rslidar_points",
        "/ir_image",
        "/gnss_pub_node/current_pose",
        "/gnss_pub_node/current_imu",
    ])
    args = parser.parse_args()
    init(args)
    bag_list = []
    for i in args.rosbag_path_list:
        bag_list.extend(sorted(glob(i + '*.bag')))
    if args.specific_rosbag!='':
        args.rosbag_list = [args.specific_rosbag]
    else:
        args.rosbag_list = bag_list
    rosbag_topic_list = []
    for bag in args.rosbag_list:
        for topic in args.topic_list:
            rosbag_topic_list.append((bag,topic))
    args.rosbag_topic_list = rosbag_topic_list

    if args.distribute:
        processes = Pool(args.processes_num)
        for rank in range(args.processes_num):
            processes.apply_async(main_work,args=(rank,args))
        processes.close()
        processes.join()
        print('dist done')
    else:
        args.processes_num = 1
        main_work(0,args)
        print('done')
