#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  create_dataset_labels.py
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
Generate datasets to be usend when training caffe. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com) 
''' 

import argparse, os
import random

def load_args():
	ap = argparse.ArgumentParser(description='Generate datasets to be usend when training caffe.')
	ap.add_argument('-i', '--frames-file-list',
					dest='input_path',
					help='path to the folder of frames.',
					type=str, required=True)
	ap.add_argument('-o', '--output-path',
					dest='output_path',
					help='path to output the dataset generated.',
					type=str, required=True)
	ap.add_argument('-t', '--train-type',
					dest='train_type',
					help='localization or classification.',
					type=str, required=True)
	ap.add_argument('-fid', '--fold-id',
					dest='fold_id',
					help='choose the fold scheme to mount.',
					type=str, required=False)
	ap.add_argument('-d', '--dataset-path',
					dest='dataset_path',
					help='inform dataset path.',
					type=str, required=True)

	args = ap.parse_args()
	
	print args
	
	return args

def get_file_list_content(filename):
	with open(filename) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	return content

	
def create_labels_by_fold(input_dir, output_dir_path, train_type, fold_id, dataset_path):
	train_list_positive_file = os.path.join(os.path.dirname(dataset_path), 'folds', fold_id, 'positive_network_training_set.txt')
	train_list_negative_file = os.path.join(os.path.dirname(dataset_path), 'folds', fold_id, 'negative_network_training_set.txt')

	validation_list_positive_file = os.path.join(os.path.dirname(dataset_path), 'folds', fold_id, 'positive_network_validation_set.txt')
	validation_list_negative_file = os.path.join(os.path.dirname(dataset_path), 'folds', fold_id, 'negative_network_validation_set.txt')

	test_list_positive_file = os.path.join(os.path.dirname(dataset_path), 'folds', fold_id + '_positive_test.txt')
	test_list_negative_file = os.path.join(os.path.dirname(dataset_path), 'folds', fold_id + '_negative_test.txt')

#	print "train_list_positive_file : %s" % train_list_positive_file
#	print "train_list_negative_file : %s" % train_list_negative_file
#	print "test_list_positive_file : %s" % test_list_positive_file
#	print "test_list_negative_file : %s" % test_list_negative_file

	train_list      = get_file_list_content(train_list_positive_file) + get_file_list_content(train_list_negative_file)
	validation_list = get_file_list_content(validation_list_positive_file) + get_file_list_content(validation_list_negative_file)
	test_list       = get_file_list_content(test_list_positive_file) + get_file_list_content(test_list_negative_file)
	
	
#	train_list.sort()
#	validation_list.sort()
#	test_list.sort()
	random.shuffle(train_list)
	random.shuffle(validation_list)
	random.shuffle(test_list)

#	print "train_list : %s" % train_list
#	print "test_list : %s" % test_list
	
	train_list_label_file = os.path.dirname(output_dir_path) + "/" + fold_id + "_train_list.label"
	validation_list_label_file = os.path.dirname(output_dir_path) + "/" + fold_id + "_validation_list.label"
	test_list_label_file = os.path.dirname(output_dir_path) + "/" + fold_id + "_test_list.label"
	if os.path.isfile(train_list_label_file):
		print "Will remove %s" % train_list_label_file
		os.remove(train_list_label_file)
	if os.path.isfile(validation_list_label_file):
		print "Will remove %s" % validation_list_label_file
		os.remove(validation_list_label_file)
	if os.path.isfile(test_list_label_file):
		print "Will remove %s" % test_list_label_file
		os.remove(test_list_label_file)

	for video in train_list:
		mount_labels(input_dir, video, train_type, train_list_label_file)

	for video in validation_list:
		mount_labels(input_dir, video, train_type, validation_list_label_file)

	for video in test_list:
		mount_labels(input_dir, video, train_type, test_list_label_file)


def mount_labels(input_dir, video, train_type, file_to_write_label):
	file_label = open(file_to_write_label, 'a')

	for (root, dirs, files) in os.walk(os.path.dirname(input_dir) + "/" + video):
		#print root
		if files:
			buffer = ""
			#print "file len %d" % len(files)
			files.sort()
			#print "files : %s" % files
			#print "sorted_files : %s" % sorted_files
			for file_name in files:
				if (train_type == 'classification'):
					if 'vPorn' in file_name:
						buffer += os.path.dirname(input_dir) + "/" + video + "/" + file_name + " 1\n"
					else:
						buffer += os.path.dirname(input_dir) + "/" + video + "/" + file_name + " 0\n"
				elif(train_type == 'localization'):
						buffer += os.path.dirname(input_dir) + "/" + video + "/" + file_name + " " + file_name.split('.')[3] + "\n"
				else:
					print "train_type invalid!!!"
					return 1
			file_label.write(buffer)
		else:
			print "no files"
	file_label.close()

def create_labels(input_dir, output_dir_path, train_type, fold_id, dataset_path):

	for (root, dirs, files) in os.walk(input_dir):
		#print root
		if files:
			fold = os.path.basename(os.path.normpath(root))
			if not os.path.isdir(output_dir_path):
				os.makedirs(output_dir_path)
			file_label_name = os.path.join(output_dir_path, fold+".label" )
			if os.path.isfile(file_label_name):
				print "Will remove %s" % file_label_name
				os.remove(file_label_name)
			file_label = open(file_label_name, 'a')
			#print file_label_name
			buffer = ""
			#print "file len %d" % len(files)
			files.sort()
			#print "files : %s" % files
			#print "sorted_files : %s" % sorted_files
			for file_name in files:
				if (train_type == 'classification'):
					if 'vPorn' in file_name:
						buffer += file_name + " 1\n"
					else:
						buffer += file_name + " 0\n"
				elif(train_type == 'localization'):
						buffer += file_name + " " + file_name.split('.')[3] + "\n"
				else:
					print "train_type invalid!!!"
					return 1
			file_label.write(buffer)
			file_label.close()
		else:
			print "no files"

def main():
	args = load_args()

	if not os.path.isdir(args.output_path):
		os.makedirs(args.output_path)
	if args.fold_id:
		create_labels_by_fold(args.input_path, args.output_path, args.train_type, args.fold_id, args.dataset_path)
	else:
		create_labels(args.input_path, args.output_path, args.train_type, args.fold_id, args.dataset_path)


if __name__ == '__main__':
	main()
