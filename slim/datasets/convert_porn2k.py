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
      'image/encoded': bytes_feature(image_data),
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


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'porn2k_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id+1, num_shards)
  return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, metadata, label_dir, dataset_dir):
  """Converts the given images and metadata to a TFRecord dataset.
  Args:
    split_name: The name of the dataset: 'train', 'validation', or 'test'
    metadata: A list with the dataset metadata
    label_dir: The directory with the input file list
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'test', 'validation', 'svm_train' , 'svm_validation']

  dataset_size = len(metadata)
  metadata = iter(metadata)
  num_shards = int(math.ceil(dataset_size / _NUM_PER_SHARD))
  if ( dataset_size % _NUM_PER_SHARD < int(_NUM_PER_SHARD/3.0) ) :
    num_shards = max(num_shards-1, 1)

  with tf.Graph().as_default(), tf.Session('') as session :
    image_reader = ImageReader()

    for shard_id in range(num_shards) :
      output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards)
      tfrecord_writer = tf.python_io.TFRecordWriter(output_filename);

      start_ndx = shard_id*_NUM_PER_SHARD
      end_ndx   = (shard_id+1)*_NUM_PER_SHARD if shard_id<num_shards-1 else dataset_size
      for i in range(start_ndx, end_ndx) :
        sys.stdout.write('\r>> Converting image %d/%d shard %d' %
          (i+1, dataset_size, shard_id))
        sys.stdout.flush()

        # Read the filename:
        meta = metadata.next()
#        image_file = os.path.join(label_dir, meta[0]) + '.jpg'
        image_file = meta[0]
        file_name = meta[0].split('/')[6].split('.')
        image_id = file_name[0] + '.' + file_name[1] + '.' + file_name[2] + '.' + file_name[3]
#        print('INFO: image_id', image_id,  file=sys.stderr) 
        image_data = tf.gfile.FastGFile(image_file, 'r').read()
        height, width = image_reader.read_image_dims(session, image_data)

#        class_name = os.path.basename(os.path.dirname(filenames[i]))
#        class_id = class_names_to_ids[class_name]
        class_id = int(meta[1])

        example = _image_to_tfexample(image_data, 'jpg', height, width, class_id, image_id)
        tfrecord_writer.write(example.SerializeToString())

      tfrecord_writer.close()
  sys.stdout.write('\n')
  sys.stdout.flush()

def get_file_list_content(filename):
	with open(filename) as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	content = [ m.split(' ') for m in content ]
	return content

def run(mode, folder_id, label_dir, dataset_dir, blacklist_file=None, excluded_scope=None) :
  """Runs the download and conversion operation.
  Args:
    mode: TRAIN or TEST
    folder_id: The name of the folder being put into dataset format
    label_dir: The directory with the input file list
    dataset_dir: The dataset directory where the dataset is stored
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  train_file = os.path.join(label_dir, folder_id + "_train_list.label") 
  validation_file = os.path.join(label_dir, folder_id + "_validation_list.label") 
  test_file = os.path.join(label_dir, folder_id + "_test_list.label") 
  train_metadata = get_file_list_content(train_file)
  validation_metadata = get_file_list_content(validation_file)
  test_metadata = get_file_list_content(test_file)
  print('INFO: train_len', len(train_metadata), file=sys.stderr)
  print('INFO: validation_len', len(validation_metadata), file=sys.stderr)
  print('INFO: test_len', len(test_metadata), file=sys.stderr)
#  # Get metadata
##  metadata = [ m.strip().split(',') for m in open(folder_id, 'r').readlines() ]
##  metadata = [ [ f.strip() for f in m] for m in metadata ]
##  metadata = metadata[1:] # Skips header
#  metadata = get_file_list_content(folder_id)
##  print('INFO: metadata ', metadata[0], file=sys.stderr)
#  metadata = [ m.split(' ') for m in metadata ]
##  print('INFO: metadata ', metadata[0], file=sys.stderr)
#
# Get blacklist
  if not blacklist_file is None :
    blacklist = [ r.strip() for r in open(blacklist_file, 'r') ]
    train_metadata  = [ m for m in train_metadata if not (m[0] in blacklist) ]
    validation_metadata  = [ m for m in validation_metadata if not (m[0] in blacklist) ]

  if mode == 'TRAIN' :
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
#    # Divide into train, validation, and test sets...
#    random.seed(_RANDOM_SEED)
#    random.shuffle(meta_porn)
#    random.shuffle(meta_nonporn)

#    # ...porn
#    n_porn       = len(meta_porn)
#    n_training   = int(n_porn/100.0*_TRAINING_PERC)
#    n_validation = int(n_porn/100.0*_VALIDATION_PERC)
#    training     = meta_porn[:n_training]
#    validation   = meta_porn[n_training:n_training+n_validation]
#    test         = meta_porn[n_training+n_validation:]
#    # ...nonporn
#    n_nonporn    = len(meta_nonporn)
#    n_training   = int(n_nonporn/100.0*_TRAINING_PERC)
#    n_validation = int(n_nonporn/100.0*_VALIDATION_PERC)
#    training    += meta_nonporn[:n_training]
#    validation  += meta_nonporn[n_training:n_training+n_validation]
#    test        += meta_nonporn[n_training+n_validation:]

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
        _convert_dataset('train', training, label_dir, dataset_dir)
    if 'validation' not in excluded_scope:
        _convert_dataset('validation', validation, label_dir, dataset_dir)
    if 'test' not in excluded_scope:
        _convert_dataset('test', test, label_dir, dataset_dir)

    # Saves classes and split sizes
    print('INFO: ', CLASSES_TO_SIZES, file=sys.stderr)
    pickle.dump(CLASSES_TO_SIZES, open(os.path.join(dataset_dir, 'classes_to_sizes.pkl'), 'w'))

    SPLITS_TO_SIZES = { 'train'      : len(training),
                        'validation' : len(validation),
                        'test'       : len(test) }
    print('INFO: ', SPLITS_TO_SIZES,  file=sys.stderr)
    pickle.dump(SPLITS_TO_SIZES,  open(os.path.join(dataset_dir, 'splits_to_sizes.pkl'),  'w'))

    # Logs splits
    open(os.path.join(dataset_dir, 'train_set.log'), 'w').write('\n'.join([m[0] for m in training])+'\n')
    open(os.path.join(dataset_dir, 'valid_set.log'), 'w').write('\n'.join([m[0] for m in validation])+'\n')
    open(os.path.join(dataset_dir, 'test_set.log'),  'w').write('\n'.join([m[0] for m in test])+'\n')

#  elif mode == 'TEST' :
#    training   = []
#    validation = []
#    test       = metadata
#    _convert_dataset('test', test, label_dir, dataset_dir)
#
#    # This data is still needed by the prediction script
#    CLASSES_TO_SIZES = { 'melanoma'  : 0,
#                         'keratosis' : 0,
#                         'other'     : len(test) }
  elif mode == 'SVM_DATA' :
    svm_train_file = os.path.join(label_dir, folder_id + "_svm_train_list.label") 
    svm_validation_file = os.path.join(label_dir, folder_id + "_svm_validation_list.label") 
    svm_train_metadata = get_file_list_content(svm_train_file)
    svm_validation_metadata = get_file_list_content(svm_validation_file)
    print('INFO: svm_train_len', len(svm_train_metadata), file=sys.stderr)
    print('INFO: svm_validation_len', len(svm_validation_metadata), file=sys.stderr)
    # Get blacklist
    if not blacklist_file is None :
      blacklist = [ r.strip() for r in open(blacklist_file, 'r') ]
      svm_train_metadata  = [ m for m in svm_train_metadata if not (m[0] in blacklist) ]
      svm_validation_metadata  = [ m for m in svm_validation_metadata if not (m[0] in blacklist) ]

    # Divide metadata into stratified sets
    svm_train_meta_porn  = [ m for m in svm_train_metadata if m[1]=='1' ]
    svm_train_meta_nonporn = [ m for m in svm_train_metadata if m[1]=='0' ]
    svm_validation_meta_porn  = [ m for m in svm_validation_metadata if m[1]=='1' ]
    svm_validation_meta_nonporn = [ m for m in svm_validation_metadata if m[1]=='0' ]
    
    svm_meta_porn = svm_train_meta_porn + svm_validation_meta_porn
    svm_meta_nonporn = svm_train_meta_nonporn + svm_validation_meta_nonporn

    CLASSES_TO_SIZES = { 'porn'  : len(svm_meta_porn),
                         'nonporn' : len(svm_meta_nonporn) }
    print('INFO: ', CLASSES_TO_SIZES, file=sys.stderr)
    svm_training   = svm_train_meta_porn + svm_train_meta_nonporn
    svm_validation = svm_validation_meta_porn + svm_validation_meta_nonporn
    
    random.seed(_RANDOM_SEED)
    random.shuffle(svm_training)
    random.shuffle(svm_validation)

    print('INFO: train_len 2', len(svm_training), file=sys.stderr)
    print('INFO: validation_len 2', len(svm_validation), file=sys.stderr)

    # Convert the training and validation sets.
    if 'svm_train' not in excluded_scope:
        _convert_dataset('svm_train', svm_training, label_dir, dataset_dir)
    if 'svm_validation' not in excluded_scope:
        _convert_dataset('svm_validation', svm_validation, label_dir, dataset_dir)

    # Saves classes and split sizes
    print('INFO: ', CLASSES_TO_SIZES, file=sys.stderr)
    pickle.dump(CLASSES_TO_SIZES, open(os.path.join(dataset_dir, 'svm_classes_to_sizes.pkl'), 'w'))

    SPLITS_TO_SIZES = { 'svm_train'      : len(svm_training),
                        'svm_validation' : len(svm_validation) }
    print('INFO: ', SPLITS_TO_SIZES,  file=sys.stderr)
    pickle.dump(SPLITS_TO_SIZES,  open(os.path.join(dataset_dir, 'svm_splits_to_sizes.pkl'),  'w'))

    # Logs splits
    open(os.path.join(dataset_dir, 'svm_train_set.log'), 'w').write('\n'.join([m[0] for m in svm_training])+'\n')
    open(os.path.join(dataset_dir, 'svm_valid_set.log'), 'w').write('\n'.join([m[0] for m in svm_validation])+'\n')

  else :
    print('FATAL: invalid mode ', mode, file=sys.stderr)
    sys.exit(1)



if __name__=='__main__' :
#  if (len(sys.argv)>4  and sys.argv[1]=='TEST') :
#    run(mode=sys.argv[1], folder_id=sys.argv[2], label_dir=sys.argv[3], dataset_dir=sys.argv[4])
#  elif (len(sys.argv)>5  and sys.argv[1]=='TRAIN') :
    print (sys.argv)
    run(mode=sys.argv[1], folder_id=sys.argv[2], label_dir=sys.argv[3], dataset_dir=sys.argv[4], blacklist_file=sys.argv[5], excluded_scope=sys.argv[6])
#  else :
#    print('For preparing split train / validation / test for trainin phase:\n\n'
#          'usage: convert_skin_lesion_dataset.py TRAIN folder_id label_dir dataset_dir blacklist_file\n\n',
#          'For preparing single file for test (without splitting or shuffling)\n'
#          'usage: convert_skin_lesion_dataset.py TEST folder_id label_dir dataset_dir', file=sys.stderr)
#    sys.exit(1)
