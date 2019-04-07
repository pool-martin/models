#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  create_splits.py
#  
#  Copyright 2018 Joao Paulo Martin <joao.paulo.pmartin@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; 
#  either version 2 of the License, or (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
#  PURPOSE.  See the GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth 
#  Floor, Boston, MA 02110-1301, USA.
#  
#  

''' 
Create Splits to be used in 2D or 3D CNN models. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com)
'''

import argparse, os, time, random, math
from subprocess import call
import decimal
from utils import opencv

def load_args():
    ap = argparse.ArgumentParser(description='Create Splits to be used in 2D or 3D CNN models.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/UCF-101/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/UCF-101/splits/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1')
    ap.add_argument('-sr', '--sample-rate',
                                    dest='sample_rate',
                                    help='sample rate to be used in video frame sampling.',
                                    type=int, required=False, default=1)
    ap.add_argument('-sl', '--snippet-length',
                                    dest='snippet_length',
                                    help='length of snippets for 3D splits in number of frames.',
                                    type=int, required=False, default=16)
    ap.add_argument('-sw', '--snippet-width',
                                    dest='snippet_width',
                                    help='time width of snippets for 3D splits in seconds.',
                                    type=int, required=False, default=1)
    ap.add_argument('-cf', '--contiguous-frames',
                                    dest='contiguous_frames',
                                    help='should be contiguous frames selected.',
                                    type=int, required=False, default=0)
    ap.add_argument('-et', '--engine-type',
                                    dest='engine_type',
                                    help='skvideo or opencv.',
                                    type=str, required=False, default='opencv')
    args = ap.parse_args()
    print(args)
    return args

def frange(x, y, jump):
  while x < y:
    yield int(round(x))
    x += jump

def loadUCF101Classes(args):
    with open(os.path.join(args.dataset_dir, 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt')) as f:
        class_list = f.read().splitlines()
    class_list = [x.split(" ") for x in class_list]
    return class_list

def defineUCF101Class(video_name, classes):
    for video_class in classes:
        if video_class[1] in video_name:
            return video_class[0]

def select_video_frames(video, split_type, args, split_test, classes):

    if split_test:
        video_name = video
        video_class = defineUCF101Class(video_name, classes)
    else:
        video_name, video_class = video.split(" ")
    print('\n', video_name, ' ', end='')
    frames = []
    frame_count, fps, _, _ = opencv.get_video_params(os.path.join(args.dataset_dir, 'videos', video_name))

    for frame_position in list(frange(0, frame_count, decimal.Decimal(str(fps * args.sample_rate)))):
        frame_entry = "{}_{}_{}".format(video_name, video_class, frame_position)
        frames.append(frame_entry)

    return frames

def create_video_split(name_set, split_type, all_set, args, split_test=False):
    split = []
    classes = loadUCF101Classes(args)
    for video in all_set:
        split.extend(select_video_frames(video.strip(), split_type, args,split_test, classes))


    split_path = os.path.join(args.output_path, args.split_number, split_type, '{}_fps'.format(args.sample_rate), args.engine_type, name_set + '.txt')
    with open(split_path, "w") as f:
        for item in split:
                f.write("%s\n" % item)

def create_splits(args):
    network_validation_set = []
    network_training_set = []

    test_set = []

    full_dir_path = os.path.join(args.dataset_dir, 'folds_art', args.split_number)

    ###########################################
    #collecting all split1 training videos

    network_training_set_path = os.path.join(full_dir_path, 'network_training_set.txt')
    with open(network_training_set_path) as f:
        network_training_set = f.readlines()

    ###########################################
    #collecting all split1 validation videos

    network_validation_set_path = os.path.join(full_dir_path, 'network_validation_set.txt')
    with open(network_validation_set_path) as f:
        network_validation_set = f.readlines()
        

    ###########################################
    #collecting all split1 test videos

    test_set_path = os.path.join(full_dir_path, 'test_set.txt')

    with open(test_set_path) as f:
        test_set = f.readlines()


    full_dir_path = os.path.join(args.output_path, args.split_number, '2D', '{}_fps'.format(args.sample_rate), args.engine_type, 'w_{}_l_{}'.format(args.snippet_width, args.snippet_length))
    command = "mkdir -p " + full_dir_path
    print('\n', command)
    call(command, shell=True)

    create_video_split('network_training_set', '2D', network_training_set, args)
    create_video_split('network_validation_set', '2D', network_validation_set, args)
    create_video_split('test_set', '2D', test_set, args, split_test=True)
    
def main():
    print('> Create splits from videos -', time.asctime( time.localtime(time.time())))
    args = load_args()
    create_splits(args)
    print('\n> Create splits  from videos done -', time.asctime( time.localtime(time.time())))
    return 0


if __name__ == '__main__':
    main()