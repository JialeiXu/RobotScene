from glob import glob
import os

root_path = '/nas/xujl/datasets/parse_data3'
scenes_list = sorted(glob(root_path+'/*'))

if os.path.exists('log.txt'):
    os.remove('log.txt')

with open('log.txt','a+') as f:
    for scene in scenes_list:
        data_list = sorted(glob(scene+'/*'))
        i=0
        for data in data_list:
            num = len(glob(data + '/*'))
            if i!=0:
                f.write(data  + ' '+str(num) + '\n')
            else:
                f.write(data  + ' '+str(num) + ' ' + str(num/16)  + '\n')
            i=1
        f.write('\n')

