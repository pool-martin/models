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

def load_args():
    ap = argparse.ArgumentParser(description='Create Splits to be used in 2D or 3D CNN models.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/UCF-101/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/UCF-101/folds_art/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='1')
    args = ap.parse_args()
    print(args)
    return args



def create_splits(args):
    network_validation_set = []
    network_training_set = []

    #collecting all split1 videos
    with open(os.path.join(args.dataset_dir, args.dataset_dir, 'UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist0{}.txt'.format(args.split_number))) as f:
            content = content = f.read().splitlines()

    #choosing the SVM set
    random.seed(a='seed', version=2)
    secure_random = random.SystemRandom()


    content = [x.split(" ") for x in content]

    ########################################################

    ######### Choosing the network validation set
    network_qty = len(content)
    print('Network qty: ', network_qty, sep='')

    for vClass in range(1, 102):
        videos = [x[0] for x in content if x[1] == str(vClass)]
        #choosing svm validation set
        while(len(videos) > (.85 * network_qty)):
                video_choosed = secure_random.choice(videos)
                network_validation_set.append(video_choosed)
                videos.remove(video_choosed)
                content.remove(video_choosed)

    #the rest is the network training set
    network_training_set = content

    full_dir_path = os.path.join(args.output_path, 's' + args.split_number)
    command = "mkdir -p " + full_dir_path
    print(command)
    call(command, shell=True)

    print("network_training_set %d" % len(network_training_set))
    print("network_validation_set %d" % len(network_validation_set))

    #creating files for network training
    network_training_set_path = os.path.join(full_dir_path, 'network_training_set.txt')
    with open(network_training_set_path, "w") as f:
            for item in network_training_set:
                    f.write("{} {}\n".format(item[0], item[1]))
    #creating files for network validation
    network_validation_set_path = os.path.join(full_dir_path, 'network_validation_set.txt')
    with open(network_validation_set_path, "w") as f:
            for item in network_validation_set:
                    f.write("{} {}\n".format(item[0], item[1]))

    command = 'cp /DL/UCF-101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist0{}.txt {}/test_set.txt'.format(args.split_number, full_dir_path)
    print('\n', command)
    call(command, shell=True)

def main():
    print('> Create splits from videos -', time.asctime( time.localtime(time.time())))
    args = load_args()
    split_path = os.path.join(args.output_path, args.split_number)
    if os.path.exists(split_path):
        print('Warning The split {} already exists in {}. We are exiting because this code can\'t reproduce the same sort previously done.\n If you need delete the split folder '.format(args.split_number, split_path))
    else:
        create_splits(args)
    print('\n> Create splits  from videos done -', time.asctime( time.localtime(time.time())))
    return 0


if __name__ == '__main__':
    main()