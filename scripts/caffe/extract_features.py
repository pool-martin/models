#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#  extract_features.py
#  
#  Copyright 2016 Mauricio Perez <mperez@mperez>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

'''
Extract features from the given images using caffe googlenet architecture.

- Authors: Mauricio Perez (mauriciolp84@gmail.com)
'''

import argparse
import os, string, subprocess, sys, os.path
from glob import glob
import numpy as np
import time
import caffe

import multiprocessing
from joblib import Parallel, delayed

import warnings

def parse_mean_binaryproto(filename):
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open( filename , 'rb' ).read()
	blob.ParseFromString(data)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	out = arr[0]
	out = np.ascontiguousarray(out.transpose(1,2,0))
	out = out.astype(np.uint8)
	
	return out

def load_args():
	ap = argparse.ArgumentParser(description='Extract features from the given images using caffe googlenet architecture.')

	ap.add_argument('-i', '--input_dir',
			dest='input_dir',
			help='path to the input directory, where the images are.',
			type=str, required=True)
	ap.add_argument('-l', '--imgs_list',
			dest='imgs_list',
			help='path to the list of imgs files to be processed.',
			type=str, required=False)
	ap.add_argument('-o', '--output_dir',
			help='path to the output directory',
			type=str, required=True)
	ap.add_argument('-p', '--proto_file',
			help='path to the prototxt file',
			type=str, required=True)
	ap.add_argument('-m', '--model_file',
			help='path to the model file, the pretrained net param',
			type=str, required=True)
	ap.add_argument('-ol', '--output_layer',
			help='name of layer to extract the features (expected a layer with flat output)',
			type=str, required=True)
	ap.add_argument('-a', '--mean_file',
			help='path to the mean file (expected a .binaryproto file)',
			type=str, required=False)
	ap.add_argument('-is', '--input_size',
			help='size of the input images [Width]x[Height] (Ex: 224x224)',
			type=str, required=False)
	ap.add_argument('-g', '--use_gpu', action='store_true',
			help='Use GPU, otherwise, use CPU', 
			default = False, required=False)
	ap.add_argument('-gi', '--gpu_id',
			help='id of the gpu to use',
			type=int, default = 0, required = False)
	ap.add_argument('-ms', '--minibatch-size',
			help='Size of minibatch [100]',
			type=int, default = 100, required = False)
	ap.add_argument('-vt', '--verbose-time',
			help='Verbose time measures',
			action='store_true', required = False)
	ap.add_argument('-fid', '--fold-id',
					dest='fold_id',
					help='choose the fold scheme to mount.',
					type=str, required=False)
	ap.add_argument('-d', '--dataset-path',
					dest='dataset_path',
					help='inform dataset path.',
					type=str, required=True)
					
	args = ap.parse_args()
	
	return args

def get_file_list_content(filename):
	with open(filename) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	return content

def mount_fold_list(input_dir, videos):
	file_list = []
	for video in videos:
		for (root, dirs, files) in os.walk(os.path.dirname(input_dir) + "/" + video):
			#print root
			if files:
				buffer = ""
				#print "file len %d" % len(files)
				files.sort()
				#print "files : %s" % files
				#print "sorted_files : %s" % sorted_files
				for file_name in files:
					file_full_path = os.path.dirname(input_dir) + "/" + video + "/" + file_name
					file_list.append(file_full_path)
			else:
				print "no files"
	return file_list

args = load_args()

# Creating output dir, if it doesnt exists already
if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

overall_input = []
if not args.imgs_list or not os.path.exists(args.imgs_list):
	if not args.dataset_path or not args.fold_id:
		print >> sys.stderr, "List of images doesn't exist"
		sys.exit(0)
	else:
		train_list_positive_file      = os.path.join(os.path.dirname(args.dataset_path), 'folds', args.fold_id, 'positive_svm_training_set.txt')
		train_list_negative_file      = os.path.join(os.path.dirname(args.dataset_path), 'folds', args.fold_id, 'negative_svm_training_set.txt')

		validation_list_positive_file = os.path.join(os.path.dirname(args.dataset_path), 'folds', args.fold_id, 'positive_svm_validation_set.txt')
		validation_list_negative_file = os.path.join(os.path.dirname(args.dataset_path), 'folds', args.fold_id, 'negative_svm_validation_set.txt')

		test_list_positive_file       = os.path.join(os.path.dirname(args.dataset_path), 'folds', args.fold_id + '_positive_test.txt')
		test_list_negative_file       = os.path.join(os.path.dirname(args.dataset_path), 'folds', args.fold_id + '_negative_test.txt')

	#	print "train_list_positive_file : %s" % train_list_positive_file
	#	print "train_list_negative_file : %s" % train_list_negative_file
	#	print "test_list_positive_file : %s" % test_list_positive_file
	#	print "test_list_negative_file : %s" % test_list_negative_file

		train_list = get_file_list_content(train_list_positive_file) + get_file_list_content(train_list_negative_file)
		validation_list = get_file_list_content(validation_list_positive_file) + get_file_list_content(validation_list_negative_file)
		test_list = get_file_list_content(test_list_positive_file) + get_file_list_content(test_list_negative_file)
		
		train_list.sort()
		validation_list.sort()
		test_list.sort()
		overall_input = mount_fold_list(args.input_dir, train_list) + mount_fold_list(args.input_dir, test_list) + mount_fold_list(args.input_dir, validation_list)
		
#		print overall_input
#		sys.exit(0)
else:
	imgs_files = open(args.imgs_list).read().split()
	overall_input = [args.input_dir + '/' + filename[2:] for filename in imgs_files]

##### Caffe options, transformer setting and network initiation BEGIN ####

if args.use_gpu:
	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)
else:
	caffe.set_mode_cpu()
	
net = caffe.Net(args.proto_file, args.model_file, caffe.TEST)

#~ net.blobs['data'].reshape(1,3,224,224)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
if args.mean_file is not None:
	mean_mat = parse_mean_binaryproto(args.mean_file)
	transformer.set_mean('data', mean_mat.mean(0).mean(0)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

##### Caffe options, transformer setting and network initiation END ####



input_size = (int(args.input_size.split('x')[0]),int(args.input_size.split('x')[1]))

nfiles = len(overall_input)
nbatches = np.ceil(nfiles / float(args.minibatch_size))

print 'A total of %d minibatches will be processed, containning at max %d images each, from a set of %d images overall' % (
		nbatches, args.minibatch_size, nfiles)

num_cores = multiprocessing.cpu_count()

for bidx, fidx in enumerate(xrange(0, nfiles, args.minibatch_size)):
	t0 = time.time()
	toprocess_input = overall_input[fidx:fidx+args.minibatch_size]

	def preprocess_img(im_f, transformer):
		#~ warnings.simplefilter('default')
		warnings.simplefilter('ignore')
		#~ warnings.simplefilter('always')
		"""
		Filter for suppressing the msg:
		/usr/local/lib/python2.7/dist-packages/scikit_image-0.11.3-py2.7-linux-x86_64.egg/skimage/external/tifffile/tifffile_local.py:3246: UserWarning: unexpected end of lzw stream (code 0)
		"""
		return transformer.preprocess('data', caffe.io.load_image(im_f))

	images_minibatch = np.array(Parallel(n_jobs=num_cores)(delayed(preprocess_img)(im_f, transformer) for im_f in toprocess_input))
	
	if args.verbose_time:
		t1 = time.time()
		print '\tReading data took %g seconds to process %d images' % (
			t1-t0, len(images_minibatch))
	
	net.blobs['data'].reshape(len(toprocess_input),3,input_size[0],input_size[1])
	
	net.blobs['data'].data[...] = images_minibatch
	
	out = net.forward()
	
	if args.verbose_time:
		t2 = time.time()
		print '\tnet.forward() took %g seconds to process %d images' % (
			t2-t1, len(images_minibatch))
		
	i = 0
	filename = ""
	f_size = 0
	
	### Process output, saving separately files ###
	for dsc_line in net.blobs[args.output_layer].data:
		if i >= len(toprocess_input):
			print  >> sys.stderr, "An Error has ocurred while processing the images"
			break

		rel_fname = os.path.relpath(toprocess_input[i], args.input_dir)
		output_fname = os.path.join(args.output_dir, rel_fname + ".dsc")
		output_fname_label = os.path.join(args.output_dir, rel_fname + ".label")
		output_dir = os.path.split(output_fname)[0]
		
		output_fname = output_fname.replace('\ ',' ')
		output_fname_label = output_fname_label.replace('\ ',' ')
		output_dir = output_dir.replace('\ ',' ')
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		
		i += 1
		
		line_label = '%d | %s\n'%(out['prob'][i].argmax(), out['prob'][i])
		line = '%d 1 1\n'%(dsc_line[:,0,0].shape[0])
		first = True
		for dsc_value in dsc_line[:,0,0]:
			if not first:
				line += ' '
			else:
				first = False
			line = line + '%.8f'%(dsc_value)
		
		f = open(output_fname, "w")
		f.write(line)
		f.close()

		f = open(output_fname_label, "w")
		f.write(line_label)
		f.close()
	
	if args.verbose_time:
		t3 = time.time()
		print '\tWriting data took %g seconds to process %d images' % (
			t3-t2, len(images_minibatch))
	
	print 'minibatch %d out of %d took %g seconds to process %d images' % (
		bidx+1, nbatches, time.time()-t0, len(toprocess_input))
	
print 'Extraction ended succesfully'
