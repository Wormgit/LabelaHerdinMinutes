import json
import glob, os
import shutil
import argparse
import cv2
from random import sample
import random
import shutil
# RGB adress.

parser = argparse.ArgumentParser()
parser.add_argument('--set_dir', default='/home/io18230/Desktop/Identification_WILL155_order/RGB/', type=str) #RGBDCows2020/Identification/RGB
#parser.add_argument('--set_dir', default='/home/io18230/Desktop/PAPER/retrain/all_distance__277q_self/RGB/')
args = parser.parse_args()
cow = {}
delete_folder=[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

n = 0
for im in sorted(os.listdir(args.set_dir)):
    cow[im] = {}
    path = os.path.join(args.set_dir,im)
    list_sub_folder = os.listdir(path)
    for item in list_sub_folder:
        cow[im][item] = {}
        list_of_file = os.listdir(os.path.join(path, item))

        #remove those that the folder only contains 1 image
        if len(list_of_file) < 2 :
            print(f'remove folder{im}')
            shutil.rmtree(path)
            break

        n += len(list_of_file)
        train, other = data_split(list_of_file, ratio=1, shuffle=True)
        val, test= data_split(other, ratio=1, shuffle=True)

        cow[im][item]['test'] = test
        cow[im][item]['train'] = train
        cow[im][item]['valid'] = val


print(f'total is {n}')
sc ='/home/io18230/Desktop/'+'single_train_valid_test_splits.json'

with open(sc, 'w+') as f:
    json.dump(cow, f)