import cv2
import os
from glob import glob
import numpy as np
from tqdm import tqdm
root_data = '/nas/xujl/datasets/parse_data/'
scene = '2024-04-18-17-29-30'
cam = 'ir'#cam0,ir
save_path = 'ir_video/'

scene_list = sorted(glob(root_data + '*'))
#print('scene_list',scene_list)

def compress_image(image_path, quality=50,size=0.5):
    # 读取图片
    image = cv2.imread(image_path)

    image = cv2.resize(image,(int(2880*size),int(1860*size)))

    # 获取图片尺寸
    height, width, _ = image.shape

    # 压缩图片
    compressed_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]#.tostring()

    return compressed_image, width, height

for scene in scene_list:
    if scene!='/data_nfs/xujl/datasets/parse_data/2024-04-30-14-50-23':continue
    print('data_path',scene + '/' + cam + '/*')
    #
    # img_list = sorted(glob(scene + '/' + cam + '/*'))
    # s=2
    # size = (2880//s,1860//s) if cam=='cam0' else (1280,1024)
    #
    # os.makedirs(save_path,exist_ok=True)
    # video_write = cv2.VideoWriter(save_path + '%s_%s_short.avi'%(scene.split('/')[-1],cam),cv2.VideoWriter_fourcc(*'MJPG'),1,size)
    #
    # #rgb_depth_array = np.zeros((2880//s,1860//s,3))
    # i=0
    # for img in tqdm(img_list[68160:68480:1]):
    #     if i==3*60+8:
    #         print(img)
    #     img = cv2.imread(img)
    #     #print(img.shape)
    #     img = cv2.resize(img, size)
    #     video_write.write(img)
    #     #break
    #     i=i+1
    # video_write.release()
