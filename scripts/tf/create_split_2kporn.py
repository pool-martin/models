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
                                    type=str, required=False, default='/DL/2kporn/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/2kporn/folds_art/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1')
    args = ap.parse_args()
    print(args)
    return args



def create_splits(args):

    operations = ['training', 'test']

    for operation in operations:

        positive_network_validation_set = []
        negative_network_validation_set = []
        positive_network_training_set = []
        negative_network_training_set = []

        #collecting all split1 videos
        with open(os.path.join(args.dataset_dir, 'folds/{}_positive_{}.txt'.format(args.split_number, operation))) as f:
                positive_content = f.readlines()
        with open(os.path.join(args.dataset_dir,'folds/{}_negative_{}.txt'.format(args.split_number, operation))) as f:
                negative_content = f.readlines()

        positive_folder_qty = len(positive_content) 
        negative_folder_qty = len(negative_content)
        print('Positive video qty: ', positive_folder_qty, ', negative video qty: ', negative_folder_qty)

        #choosing the training and validation set
        random.seed(a='seed', version=2)
        secure_random = random.SystemRandom()

        ########################################################
        while((len(positive_content) + len(negative_content)) > (.85 * (positive_folder_qty + negative_folder_qty))):
                positive_video_choosed = secure_random.choice(positive_content)
                positive_network_validation_set.append(positive_video_choosed)
                positive_content.remove(positive_video_choosed)

                negative_video_choosed = secure_random.choice(negative_content)
                negative_network_validation_set.append(negative_video_choosed)
                negative_content.remove(negative_video_choosed)
        ########################################################

        #the rest is the network training set
        positive_network_training_set = positive_content
        negative_network_training_set = negative_content

        split_extension = 'a' if operation == 'training' else 'b'
        full_dir_path = os.path.join(args.output_path, '{}_{}'.format(args.split_number, split_extension))
        command = "mkdir -p " + full_dir_path
        print(command)
        call(command, shell=True)

        print("positive_network_training_set %d" % len(positive_network_training_set))
        print("negative_network_training_set %d" % len(negative_network_training_set))
        print("positive_network_validation_set %d" % len(positive_network_validation_set))
        print("negative_network_validation_set %d" % len(negative_network_validation_set))

        print("total network %d" % (len(positive_network_training_set) + len(negative_network_training_set) + len(positive_network_validation_set) + len(negative_network_validation_set)))

        print("qty network: %d, positive: %d [%f], negative %d [%f]" \
        %(len(positive_network_training_set) + len(negative_network_training_set), 
        len(positive_network_training_set), 
        (100.0* len(positive_network_training_set))/len(positive_network_training_set + negative_network_training_set),
        len(negative_network_training_set),
        (100.0* len(negative_network_training_set))/len(positive_network_training_set + negative_network_training_set)))

        #creating files for network training
        positive_network_training_set_path = os.path.join(full_dir_path, 'positive_network_training_set.txt')
        with open(positive_network_training_set_path, "w") as f:
                for item in positive_network_training_set:
                        f.write("%s" % item)
        negative_network_training_set_path = os.path.join(full_dir_path, 'negative_network_training_set.txt')
        with open(negative_network_training_set_path, "w") as f:
                for item in negative_network_training_set:
                        f.write("%s" % item)
        #creating files for network validation
        positive_network_validation_set_path = os.path.join(full_dir_path, 'positive_network_validation_set.txt')
        with open(positive_network_validation_set_path, "w") as f:
                for item in positive_network_validation_set:
                        f.write("%s" % item)
        negative_network_validation_set_path = os.path.join(full_dir_path, 'negative_network_validation_set.txt')
        with open(negative_network_validation_set_path, "w") as f:
                for item in negative_network_validation_set:
                        f.write("%s" % item)

        options = ['positive', 'negative']
        test_set = 'test' if operation == 'training' else 'training'
        for option in options:
                from_file = os.path.join(args.dataset_dir, 'folds/{}_{}_{}.txt'.format(args.split_number, option, test_set))
                to_file   = os.path.join(full_dir_path, '{}_test.txt'.format(option))
                command = "cp {} {}".format(from_file, to_file)
                print(command)
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