#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, csv, time, glob, shutil, copy, random
import matplotlib.pyplot as plt
import sys, math, warnings
import argparse
import json
import shutil
#tf
#import keras
#import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", default='/home/io18230/Desktop/C_annotator.csv', type=str) # ceiling_wacv C_annotator
#parser.add_argument("--csv_path", default='/home/io18230/Desktop/ceiling_wacv.csv', type=str)
parser.add_argument("--img_path", default='/home/io18230/Desktop/RGB (copy)', type=str)
args = parser.parse_args()

show_move_file = 0

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

csvFile = csv.reader(open(args.csv_path, 'r'), delimiter=',')
reader = list(csvFile)
del reader[0]
firstforder = []
secondforder = []
for i in range(len(reader)):
    firstforder.append(reader[i][2])
    secondforder.append(reader[i][3])


count_c = 0
for item in sorted(os.listdir(args.img_path)):
    #3 class
    for subitm in sorted(os.listdir(os.path.join(args.img_path,item))):
        pathName=item+'/'+subitm
        if pathName in firstforder and os.path.exists(os.path.join(args.img_path, pathName)):
            for index in range(len(firstforder)):
                if pathName == firstforder[index]:
                    for isecond in eval(secondforder[index]):
                        if pathName == isecond:
                            continue
                        if int(pathName[:3]) < int(isecond[:3]):
                            largerF = isecond
                            smallF = pathName
                        else:
                            largerF = pathName
                            smallF = isecond
                        if os.path.exists(os.path.join(args.img_path, smallF)): # 小的在,ok
                            if os.path.exists(os.path.join(args.img_path, largerF)):
                                for images in os.listdir(os.path.join(args.img_path,largerF)):
                                    sr = os.path.join(args.img_path,largerF,images)
                                    de = os.path.join(args.img_path,smallF)
                                    shutil.move(sr,de)
                                if show_move_file:
                                    print(f'Moved file from {largerF} to {smallF} and deleted {largerF}')
                                os.rmdir(os.path.join(args.img_path,largerF))
                                count_c +=1
                            else:
                                if show_move_file:
                                    print(f'{largerF} has been deleted')
                        else: #不在,判断大的是否存在
                            if os.path.exists(os.path.join(args.img_path, smallF)): #不存在,没事, if exists:
                                print(f'!!!!!pay attention to folder {largerF} and {smallF}')

                        #move files
print(f'\n*******merged {count_c} subfolders*******\n')
for item in sorted(os.listdir(args.img_path)):
    count = 0
    if not os.listdir(os.path.join(args.img_path,item)):
    #for subitm in sorted(os.listdir(os.path.join(args.img_path,item))):
        count +=1
    #if count == 0:
        os.rmdir(os.path.join(args.img_path,item))
        #print(f'deleted blank dir {item}')