# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle
import random
import sys

import tensorflow as tf


import argparse, os, time, random, math
from subprocess import call

def load_args():
    ap = argparse.ArgumentParser(description='Create tfrecord to be used in 2D or 3D CNN models.')
    ap.add_argument('-m', '--mode',
                                    dest='mode',
                                    help='modo de criação.',
                                    type=str, required=False, default='TRAIN')
    ap.add_argument('-e', '--excluded-scope',
                                    dest='excluded_scope',
                                    help='excluded scope.',
                                    type=str, required=False, default="None")
    ap.add_argument('-b', '--blacklist-file',
                                    dest='blacklist_file',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default=None)
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='where the frames are.',
                                    type=str, required=False, default='/DL/2kporn/frames/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/2kporn/tfrecords/')
    ap.add_argument('-l', '--label-dir',
                                    dest='label_dir',
                                    help='labels dir.',
                                    type=str, required=False, default='/Exp/2kporn/splits/3D')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1_a')
    ap.add_argument('-gpu', '--gpu-to-use',
                                    dest='gpu_to_use',
                                    help='gpu to use.',
                                    type=str, required=False, default=None)

    args = ap.parse_args()
    print(args)
    return args


#from datasets import dataset_utils

# Copied from deleted dataset_utils.py ===>
def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _image_to_tfexample(image_data, image_format, height, width, class_id, image_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded_0': bytes_feature(image_data[0]),
      'image/encoded_1': bytes_feature(image_data[1]),
      'image/encoded_2': bytes_feature(image_data[2]),
      'image/encoded_3': bytes_feature(image_data[3]),
      'image/encoded_4': bytes_feature(image_data[4]),
      'image/encoded_5': bytes_feature(image_data[5]),
      'image/encoded_6': bytes_feature(image_data[6]),
      'image/encoded_7': bytes_feature(image_data[7]),
      'image/encoded_8': bytes_feature(image_data[8]),
      'image/encoded_9': bytes_feature(image_data[9]),
      'image/encoded_10': bytes_feature(image_data[10]),
      'image/encoded_11': bytes_feature(image_data[11]),
      'image/encoded_12': bytes_feature(image_data[12]),
      'image/encoded_13': bytes_feature(image_data[13]),
      'image/encoded_14': bytes_feature(image_data[14]),
      'image/encoded_15': bytes_feature(image_data[15]),
      'image/encoded_16': bytes_feature(image_data[16]),
      'image/encoded_17': bytes_feature(image_data[17]),
      'image/encoded_18': bytes_feature(image_data[18]),
      'image/encoded_19': bytes_feature(image_data[19]),
      'image/encoded_20': bytes_feature(image_data[20]),
      'image/encoded_21': bytes_feature(image_data[21]),
      'image/encoded_22': bytes_feature(image_data[22]),
      'image/encoded_23': bytes_feature(image_data[23]),
      'image/encoded_24': bytes_feature(image_data[24]),
      'image/encoded_25': bytes_feature(image_data[25]),
      'image/encoded_26': bytes_feature(image_data[26]),
      'image/encoded_27': bytes_feature(image_data[27]),
      'image/encoded_28': bytes_feature(image_data[28]),
      'image/encoded_29': bytes_feature(image_data[29]),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/meta/id': bytes_feature(image_id),
  }))

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of elements per shard
_NUM_PER_SHARD = 1024

## Fractions for training, validation, and testing split
#_TRAINING_PERC   = 50
#_VALIDATION_PERC = 15
#_TESTING_PERC    = 35
#
#assert (_TRAINING_PERC + _VALIDATION_PERC + _TESTING_PERC) == 100


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_dataset_filename(output_path, split_name, shard_id, num_shards):
  output_filename = 'porn2k_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id+1, num_shards)
  return os.path.join(output_path, output_filename)

def load_fragment(splits_dir, dataset_dir, split_number, video_name, meta):
      # /Exp/2kporn/splits/3D/s1_a/3D/1_fps/opencv/w_1_l_30/vPorn000999/vPorn000999_0_6575.txt
      image_reader = ImageReader()
      image_id = '{}_{}_{}'.format(meta[0], meta[1], meta[2])
      image_name = '{}.jpg'.format(image_id)
      image_file = os.path.join(dataset_dir, video_name, image_name)

      fragment_file = os.path.join(splits_dir, 'w_1_l_30', video_name, '{}.txt'.format(image_id))

      with open(fragment_file) as f:
        content = f.read().splitlines()

      frames = ['{}_{}_{}.jpg'.format(meta[0], meta[1], x) for x in content]

      image_frames = []
      for frame in frames:
        image_file = os.path.join(dataset_dir, video_name, frame)
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        image_frames.append(image_data)

      return image_frames


def _convert_dataset(split_name, metadata, dataset_dir, output_path, splits_dir, split_number):
  """Converts the given images and metadata to a TFRecord dataset.
  Args:
    split_name: The name of the dataset: 'train', 'validation', or 'test'
    metadata: A list with the dataset metadata
    label_dir: The directory with the input file list
    output_path: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'test', 'validation']

  dataset_size = len(metadata)
  metadata = iter(metadata)
  num_shards = int(math.ceil(dataset_size / _NUM_PER_SHARD))
  if ( dataset_size % _NUM_PER_SHARD < int(_NUM_PER_SHARD/3.0) ) :
    num_shards = max(num_shards-1, 1)

  session_config = tf.ConfigProto(
      log_device_placement = False,
      allow_soft_placement = False)
  session_config.gpu_options.allow_growth=True

  with tf.Graph().as_default(), tf.Session(config=session_config) as session :
    image_reader = ImageReader()

    for shard_id in range(num_shards) :
      #  /DL/2kporn/tfrecords/s1_a/train/porn2k_train_00001-of-00204.tfrecord
      output_filename = _get_dataset_filename(output_path, split_name, shard_id, num_shards)
      tfrecord_writer = tf.python_io.TFRecordWriter(output_filename);

      start_ndx = shard_id*_NUM_PER_SHARD
      end_ndx   = (shard_id+1)*_NUM_PER_SHARD if shard_id<num_shards-1 else dataset_size
      for i in range(start_ndx, end_ndx) :
        sys.stdout.write('\r>> Converting image %d/%d shard %d' %
          (i+1, dataset_size, shard_id))
        sys.stdout.flush()

        # Read the filename:
        meta = next(metadata)
        video_name = meta[0]
        image_data = load_fragment(splits_dir, dataset_dir, split_number, video_name, meta)

        image_id = '{}_{}_{}'.format(meta[0], meta[1], meta[2])
#         image_name = '{}.jpg'.format(image_id)
#         image_file = os.path.join(dataset_dir, video_name, image_name)
# #        print('INFO: image_id', image_id,  file=sys.stderr) 
#         image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        height, width = image_reader.read_image_dims(session, image_data[0])

#        class_name = os.path.basename(os.path.dirname(filenames[i]))
#        class_id = class_names_to_ids[class_name]
        class_id = int(meta[1])

        example = _image_to_tfexample(image_data, b'jpg', height, width, class_id, image_id.encode())
        tfrecord_writer.write(example.SerializeToString())

      tfrecord_writer.close()
  sys.stdout.write('\n')
  sys.stdout.flush()

def get_file_list_content(filename):
  with open(filename) as f:
    content = f.read().splitlines()
  content = [ m.split('_') for m in content ]
  return content

def run(mode, split_number, label_dir, dataset_dir, output_path, blacklist_file=None, excluded_scope=None) :
  """Runs the download and conversion operation.
  Args:
    mode: TRAIN or TEST
    split_number: The name of the folder being put into dataset format
    label_dir: The directory with the input file list
    output_path: The dataset directory where the dataset is stored
  """
  output_dir = os.path.join(output_path, split_number)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  # /work/jp/Exp/2kporn/splits/s1_a/2D/1_fps/opencv/
  # /Exp/2kporn/splits/3D/s1_a/3D/1_fps/opencv/
  # network_training_set.txt  network_validation_set.txt  test_set.txt
  splits_dir = os.path.join(label_dir, split_number, '3D', '1_fps', 'opencv')
  train_file      = os.path.join(splits_dir, 'network_training_set.txt') 
  validation_file = os.path.join(splits_dir, 'network_validation_set.txt') 
  test_file       = os.path.join(splits_dir, 'test_set.txt') 

  train_metadata = get_file_list_content(train_file)
  validation_metadata = get_file_list_content(validation_file)
  test_metadata = get_file_list_content(test_file)
  print('INFO: train_len', len(train_metadata), file=sys.stderr)
  print('INFO: validation_len', len(validation_metadata), file=sys.stderr)
  print('INFO: test_len', len(test_metadata), file=sys.stderr)

# Get blacklist
  if not blacklist_file is None :
    blacklist = [ r.strip() for r in open(blacklist_file, 'r') ]
    train_metadata  = [ m for m in train_metadata if not (m[0] in blacklist) ]
    validation_metadata  = [ m for m in validation_metadata if not (m[0] in blacklist) ]

  if args.mode == 'TRAIN' :
    # Divide metadata into stratified sets
    train_meta_porn  = [ m for m in train_metadata if m[1]=='1' ]
    train_meta_nonporn = [ m for m in train_metadata if m[1]=='0' ]
    validation_meta_porn  = [ m for m in validation_metadata if m[1]=='1' ]
    validation_meta_nonporn = [ m for m in validation_metadata if m[1]=='0' ]
    test_meta_porn  = [ m for m in test_metadata if m[1]=='1' ]
    test_meta_nonporn = [ m for m in test_metadata if m[1]=='0' ]
    
    meta_porn = train_meta_porn + validation_meta_porn + test_meta_porn
    meta_nonporn = train_meta_nonporn + validation_meta_nonporn + test_meta_nonporn

    CLASSES_TO_SIZES = { 'porn'  : len(meta_porn),
                         'nonporn' : len(meta_nonporn) }
    print('INFO: ', CLASSES_TO_SIZES, file=sys.stderr)

    training   = train_meta_porn + train_meta_nonporn
    validation = validation_meta_porn + validation_meta_nonporn
    test       = test_meta_porn + test_meta_nonporn
    
    random.seed(_RANDOM_SEED)
    random.shuffle(training)
    random.shuffle(validation)
    random.shuffle(test)

    print('INFO: train_len 2', len(training), file=sys.stderr)
    print('INFO: validation_len 2', len(validation), file=sys.stderr)
    print('INFO: test_len 2', len(test), file=sys.stderr)

    # Convert the training and validation sets.
    if 'train' not in excluded_scope:
        _convert_dataset('train', training, dataset_dir, output_dir, splits_dir, split_number)
    if 'validation' not in excluded_scope:
        _convert_dataset('validation', validation, dataset_dir, output_dir, splits_dir, split_number)
    if 'test' not in excluded_scope:
        _convert_dataset('test', test, dataset_dir, output_dir, splits_dir, split_number)

    # Saves classes and split sizes
    print('INFO: ', CLASSES_TO_SIZES, file=sys.stderr)
    pickle.dump(CLASSES_TO_SIZES, open(os.path.join(output_dir, 'classes_to_sizes.pkl'), 'wb'))

    SPLITS_TO_SIZES = { 'train'      : len(training),
                        'validation' : len(validation),
                        'test'       : len(test) }
    print('INFO: ', SPLITS_TO_SIZES,  file=sys.stderr)
    pickle.dump(SPLITS_TO_SIZES,  open(os.path.join(output_dir, 'splits_to_sizes.pkl'),  'wb'))

    # Logs splits
    open(os.path.join(output_dir, 'train_set.log'), 'w').write('\n'.join([m[0] for m in training])+'\n')
    open(os.path.join(output_dir, 'valid_set.log'), 'w').write('\n'.join([m[0] for m in validation])+'\n')
    open(os.path.join(output_dir, 'test_set.log'),  'w').write('\n'.join([m[0] for m in test])+'\n')

  else :
    print('FATAL: invalid mode ', mode, file=sys.stderr)
    sys.exit(1)



if __name__=='__main__' :
  args = load_args()

  if args.gpu_to_use:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_to_use

  run(mode=args.mode, split_number=args.split_number, label_dir=args.label_dir, dataset_dir=args.dataset_dir, output_path=args.output_path, blacklist_file=args.blacklist_file, excluded_scope=args.excluded_scope)
