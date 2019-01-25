#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  create_splits.py
#  
#  Copyright 2017 Joao Paulo Martin <joao.paulo.pmartin@gmail.com>
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
Create Splits to be used in torch scheme. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com) 
''' 

import argparse, os, time, random
from joblib import Parallel, delayed  
import multiprocessing
from subprocess import call

def load_args():
	ap = argparse.ArgumentParser(description='Extract rgb frames from video list.')
	ap.add_argument('-s', '--split_number',
					dest='split_number',
					help='split to be created.',
					type=str, required=False, default='s1')
	ap.add_argument('-o', '--output-path',
					dest='output_path',
					help='path to output the extracted frames.',
					type=str, required=False, default='/home/jp/DL/2kporn/folds/')
	args = ap.parse_args()
	
	print args
	
	return args

def get_video_patches(folder, video_class, video_name):
	video_patches = []
	for file in os.listdir(os.path.join(folder, video_class)):
		if video_name in file:
			video_patches.append(file)
	return video_patches
		
def create_splits(args):
	positive_network_set = []
	negative_network_set = []
	positive_network_validation_set = []
	negative_network_validation_set = []
	positive_network_training_set = []
	negative_network_training_set = []

	positive_svm_set = []
	negative_svm_set = []
	positive_svm_validation_set = []
	negative_svm_validation_set = []
	positive_svm_training_set = []
	negative_svm_training_set = []

	#collecting all split1 videos
	with open('/home/jp/DL/2kporn/folds/s1_negative_training.txt') as f:
		positive_content = f.readlines()
	with open('/home/jp/DL/2kporn/folds/s1_positive_training.txt') as f:
		negative_content = f.readlines()
	
	positive_folder_qty = len(positive_content) 
	negative_folder_qty = len(negative_content)
	print 'Positive qty: %d, negative qty: %d' % (positive_folder_qty, negative_folder_qty)
	
	#choosing the SVM set
	secure_random = random.SystemRandom()

	########################################################
	while((len(positive_content) + len(negative_content)) > (.8 * (positive_folder_qty + negative_folder_qty))):
		positive_video_choosed = secure_random.choice(positive_content)
		positive_svm_set.append(positive_video_choosed)
		positive_content.remove(positive_video_choosed)

		negative_video_choosed = secure_random.choice(negative_content)
		negative_svm_set.append(negative_video_choosed)
		negative_content.remove(negative_video_choosed)
	########################################################
	
	
	positive_svm_qty = len(positive_svm_set)
	negative_svm_qty = len(negative_svm_set)
	print 'Positive svm qty: %d, negative svm qty: %d' % (positive_svm_qty, negative_svm_qty)
	#choosing svm validation set
	########################################################
	while((len(positive_svm_set) + len(negative_svm_set)) > (.85 * (positive_svm_qty + negative_svm_qty))):
		positive_video_choosed = secure_random.choice(positive_svm_set)
		positive_svm_validation_set.append(positive_video_choosed)
		positive_svm_set.remove(positive_video_choosed)

		negative_video_choosed = secure_random.choice(negative_svm_set)
		negative_svm_validation_set.append(negative_video_choosed)
		negative_svm_set.remove(negative_video_choosed)

	#the rest is the svm training set
	positive_svm_training_set = positive_svm_set
	negative_svm_training_set = negative_svm_set
	########################################################
	
	
	######### Choosing the network validation set
	positive_network_qty = len(positive_content)
	negative_network_qty = len(negative_content)
	print 'Positive network qty: %d, negative network qty: %d' % (positive_network_qty, negative_network_qty)

	#choosing svm validation set
	while((len(positive_content) + len(negative_content)) > (.85 * (positive_network_qty + negative_network_qty))):
		positive_video_choosed = secure_random.choice(positive_content)
		positive_network_validation_set.append(positive_video_choosed)
		positive_content.remove(positive_video_choosed)

		negative_video_choosed = secure_random.choice(negative_content)
		negative_network_validation_set.append(negative_video_choosed)
		negative_content.remove(negative_video_choosed)
		
	#the rest is the network training set
	positive_network_training_set = positive_content
	negative_network_training_set = negative_content

	full_dir_path = os.path.join(args.output_path, args.split_number)
	command = "mkdir -p " + full_dir_path
	print command
	call(command, shell=True)

	print "positive_network_training_set %d" % len(positive_network_training_set)
	print "negative_network_training_set %d" % len(negative_network_training_set)
	print "positive_network_validation_set %d" % len(positive_network_validation_set)
	print "negative_network_validation_set %d" % len(negative_network_validation_set)
	print "positive_svm_training_set %d" % len(positive_svm_training_set)
	print "negative_svm_training_set %d" % len(negative_svm_training_set)
	print "positive_svm_validation_set %d" % len(positive_svm_validation_set)
	print "negative_svm_validation_set %d" % len(negative_svm_validation_set)

	print "total network %d" % (len(positive_network_training_set) + len(negative_network_training_set) + len(positive_network_validation_set) + len(negative_network_validation_set))
	print "total svm %d" % (len(positive_svm_training_set) + len(negative_svm_training_set) + len(positive_svm_validation_set) + len(negative_svm_validation_set))
	
	print "qty network: %d, positive: %d [%f], negative %d [%f]" \
	%(len(positive_network_training_set) + len(negative_network_training_set), 
	len(positive_network_training_set), 
	(100.0* len(positive_network_training_set))/len(positive_network_training_set + negative_network_training_set),
	len(negative_network_training_set),
	(100.0* len(negative_network_training_set))/len(positive_network_training_set + negative_network_training_set))

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
	#creating files for svm training
	positive_svm_training_set_path = os.path.join(full_dir_path, 'positive_svm_training_set.txt')
	with open(positive_svm_training_set_path, "w") as f:
		for item in positive_svm_training_set:
			f.write("%s" % item)
	negative_svm_training_set_path = os.path.join(full_dir_path, 'negative_svm_training_set.txt')
	with open(negative_svm_training_set_path, "w") as f:
		for item in negative_svm_training_set:
			f.write("%s" % item)
	#creating files for svm validation
	positive_svm_validation_set_path = os.path.join(full_dir_path, 'positive_svm_validation_set.txt')
	with open(positive_svm_validation_set_path, "w") as f:
		for item in positive_svm_validation_set:
			f.write("%s" % item)
	negative_svm_validation_set_path = os.path.join(full_dir_path, 'negative_svm_validation_set.txt')
	with open(negative_svm_validation_set_path, "w") as f:
		for item in negative_svm_validation_set:
			f.write("%s" % item)
	
def main():
	args = load_args()
	
	print '> Ceate splits from videos -', time.asctime( time.localtime(time.time()) )
	
	create_splits(args)
	
	print '\n> Ceate splits  from videos done -', time.asctime( time.localtime(time.time()) )
	
	return 0


if __name__ == '__main__':
	main()
