# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'ucf101_%s_*.tfrecord'

#SPLITS_TO_SIZES = {'train': 3320, 'validation': 350}

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'id'       : 'id of the image.',
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 (nonPorn) and 1 (porn)',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  print('===================================')
  print('SPLIT', split_name)  
  print('===================================')
  SPLITS_TO_SIZES  = pickle.load(open(os.path.join(dataset_dir, 'splits_to_sizes.pkl'),  'rb'))
  CLASSES_TO_SIZES = pickle.load(open(os.path.join(dataset_dir, 'classes_to_sizes.pkl'), 'rb'))

  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/meta/id'         : tf.FixedLenFeature([], tf.string,  default_value=''),
      }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'id'       : slim.tfexample_decoder.Tensor('image/meta/id'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = {
    1 : 'ApplyEyeMakeup',
    2 : 'ApplyLipstick',
    3 : 'Archery',
    4 : 'BabyCrawling',
    5 : 'BalanceBeam',
    6 : 'BandMarching',
    7 : 'BaseballPitch',
    8 : 'Basketball',
    9 : 'BasketballDunk',
    10 : 'BenchPress',
    11 : 'Biking',
    12 : 'Billiards',
    13 : 'BlowDryHair',
    14 : 'BlowingCandles',
    15 : 'BodyWeightSquats',
    16 : 'Bowling',
    17 : 'BoxingPunchingBag',
    18 : 'BoxingSpeedBag',
    19 : 'BreastStroke',
    20 : 'BrushingTeeth',
    21 : 'CleanAndJerk',
    22 : 'CliffDiving',
    23 : 'CricketBowling',
    24 : 'CricketShot',
    25 : 'CuttingInKitchen',
    26 : 'Diving',
    27 : 'Drumming',
    28 : 'Fencing',
    29 : 'FieldHockeyPenalty',
    30 : 'FloorGymnastics',
    31 : 'FrisbeeCatch',
    32 : 'FrontCrawl',
    33 : 'GolfSwing',
    34 : 'Haircut',
    35 : 'Hammering',
    36 : 'HammerThrow',
    37 : 'HandstandPushups',
    38 : 'HandstandWalking',
    39 : 'HeadMassage',
    40 : 'HighJump',
    41 : 'HorseRace',
    42 : 'HorseRiding',
    43 : 'HulaHoop',
    44 : 'IceDancing',
    45 : 'JavelinThrow',
    46 : 'JugglingBalls',
    47 : 'JumpingJack',
    48 : 'JumpRope',
    49 : 'Kayaking',
    50 : 'Knitting',
    51 : 'LongJump',
    52 : 'Lunges',
    53 : 'MilitaryParade',
    54 : 'Mixing',
    55 : 'MoppingFloor',
    56 : 'Nunchucks',
    57 : 'ParallelBars',
    58 : 'PizzaTossing',
    59 : 'PlayingCello',
    60 : 'PlayingDaf',
    61 : 'PlayingDhol',
    62 : 'PlayingFlute',
    63 : 'PlayingGuitar',
    64 : 'PlayingPiano',
    65 : 'PlayingSitar',
    66 : 'PlayingTabla',
    67 : 'PlayingViolin',
    68 : 'PoleVault',
    69 : 'PommelHorse',
    70 : 'PullUps',
    71 : 'Punch',
    72 : 'PushUps',
    73 : 'Rafting',
    74 : 'RockClimbingIndoor',
    75 : 'RopeClimbing',
    76 : 'Rowing',
    77 : 'SalsaSpin',
    78 : 'ShavingBeard',
    79 : 'Shotput',
    80 : 'SkateBoarding',
    81 : 'Skiing',
    82 : 'Skijet',
    83 : 'SkyDiving',
    84 : 'SoccerJuggling',
    85 : 'SoccerPenalty',
    86 : 'StillRings',
    87 : 'SumoWrestling',
    88 : 'Surfing',
    89 : 'Swing',
    90 : 'TableTennisShot',
    91 : 'TaiChi',
    92 : 'TennisSwing',
    93 : 'ThrowDiscus',
    94 : 'TrampolineJumping',
    95 : 'Typing',
    96 : 'UnevenBars',
    97 : 'VolleyballSpiking',
    98 : 'WalkingWithDog',
    99 : 'WallPushups',
    100 : 'WritingOnBoard',
    101 : 'YoYo'
  }

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
