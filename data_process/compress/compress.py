import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Process
import argparse
import cv2
from datetime import datetime
def main_work_cam0(rank,args,png_list):
    for png_i in range(rank,png_list.__len__(),args.processes_num):
        png = png_list[png_i]
        jpg = png.replace('cam0', 'cam0_compressed').replace('png', 'jpg')
        image = cv2.imread(png)
        jpeg_quality = 95
        cv2.imwrite(jpg, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        #cmd = 'convert %s -quality 95 %s' % (png, jpg)
        #os.system(cmd)
def main_work_cam3(rank,args,png_list):
    for png_i in range(rank,png_list.__len__(),args.processes_num):
        png = png_list[png_i]
        jpg = png.replace('cam3', 'cam3_compressed').replace('png', 'jpg')
        image = cv2.imread(png)
        jpeg_quality = 95
        cv2.imwrite(jpg, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])


        #cmd = 'convert %s -quality 95 %s' % (png, jpg)
        #print(cmd)
        #os.system(cmd)

def init(args):
    if os.path.exists(args.log):
        os.remove(args.log)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.log,'a+') as f:
        f.write('Timestamp: ' + current_time + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes_num',default=96, type=int)
    parser.add_argument('--distribute',default=False, type=bool)
    parser.add_argument('--path',default='/nas/xujl/datasets/parse_data/2*', type=str)
    parser.add_argument('--log',default='./log.txt',type=str)
    args = parser.parse_args()
    init(args)

    data_list = sorted(glob(args.path))
    for scene in data_list:
        #if scene.split('/')[-1]!='2024-04-18-15-37-36': continue
        print(scene)
        path = scene + '/cam0_compressed'
        os.makedirs(path,exist_ok=True)
        path = scene + '/cam3_compressed'
        os.makedirs(path,exist_ok=True)

        png_list = sorted(glob(scene + '/cam0/*'))
        processes = Pool(args.processes_num)
        try:
            for rank in range(args.processes_num):
                processes.apply_async(main_work_cam0, args=(rank, args, png_list))
            processes.close()
            processes.join()
        except:
            with open(args.log, 'a+') as f:
                f.write(scene + ' cam0' + '\n')

        try:
            png_list = sorted(glob(scene + '/cam3/*'))
            processes = Pool(args.processes_num)
            for rank in range(args.processes_num):
                processes.apply_async(main_work_cam3, args=(rank, args, png_list))
            processes.close()
            processes.join()
        except:
            with open(args.log, 'a+') as f:
                f.write(scene + ' cam3' + '\n')
        print('done')


