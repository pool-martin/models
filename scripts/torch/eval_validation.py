#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  get_dimensions.py
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
generate_sliding_test_clips. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com) 
''' 

dataRoot = '/data/torch/ltc/datasets/2kporn/'
targetTestDir = 'validation'
import os, sys, time, argparse
import math
from subprocess import call

def load_args():
	ap = argparse.ArgumentParser(description='Create test split.')
	ap.add_argument('-i', '--inicio',
					dest='inicio',
					help='inicio.',
					type=int, required=False, default=1)
	ap.add_argument('-f', '--fim',
					dest='fim',
					help='fim.',
					type=int, required=False, default=26)
	ap.add_argument('-s', '--slide',
					dest='slide',
					help='slide.',
					type=int, required=False, default=80)
	ap.add_argument('-exp', '--experimento',
					dest='exp',
					help='experimento.',
					type=str, required=False, default="2kporn_rgb_80f_d5_center_crop")
	args = ap.parse_args()

	
	print(args)
	
	return args

def get_dir_content(folder):
	video_patches = []
	for file in os.listdir(os.path.join(folder)):
		video_patches.append(file)
	return video_patches

def eval_validation(args):

    for i in range(args.inicio,args.fim+1):
        command = "th main.lua -nFrames 80 -stream rgb -expName %s -dataset 2kporn  -dropout 0.5 -batchSize 1 -cropbeforeresize -evaluate -modelNo %d -slide %d -time_window 0 -framestep 1" % (args.exp, i, args.slide)
        print(command)
        call(command, shell=True)

def main():
    args = load_args()

    print('> Eval validation split from videos -', time.asctime( time.localtime(time.time()) ))

    eval_validation(args)

    print('\n> Eval validation split  from videos done -', time.asctime( time.localtime(time.time()) ))

    return 0


if __name__ == '__main__':
    main()
