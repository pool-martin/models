# Copyright 2017 Eduardo Valle. All rights reserved.
# eduardovalle.com/ github.com/learningtitans
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import pickle
import sys
import os

import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sk
import sklearn.decomposition
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.preprocessing

from svm_layer import utils as su

os.environ['JOBLIB_TEMP_FOLDER'] = "~/tmp"

def read_pmsvm_data(input_training):
    ids =[]
    num_samples = 0
    with open(input_training, 'r') as file:
        num_samples = sum(1 for _ in file)
    print('num_samples :',  num_samples, file=sys.stderr)
#    labels = np.empty([251616], dtype=np.float)
#    features = np.empty([251616, 1024], dtype=np.float)
#    labels = np.empty([258915], dtype=np.float)
#    features = np.empty([258915, 1024], dtype=np.float)
    labels = np.empty([num_samples], dtype=np.float)
    features = np.empty([num_samples, 1024], dtype=np.float)
#    labels = {}
#    features = {}
    i = 0
    with open(input_training, 'r') as f:
        for line in f:
            feature_dic = {}
            ids.append(i)
            feature_line = line.split(' ')
            label = int(feature_line.pop(0))
            if '\n' == feature_line[-1]:
#                feature_line = feature_line[:-1]
                feature_line.pop(-1)
            labels[i] = label
            
            for column in feature_line:
                column_split = column.split(':')
                index = long(column_split[0])
                value = float(column_split[1])
                feature_dic[index] = value
            row = np.empty([1024], dtype=np.float)
            for j in range(1, 1025):
                value = feature_dic.get(j, 0.)
                if np.isnan(value):
                    row[j-1] = 0
                else:
                    row[j-1] = value
            features[i] = row
            if(i % 1000 == 0):
                print('reading i=', i, file=sys.stderr)
            i += 1
            
    return ids, labels, features

parser = argparse.ArgumentParser(prog='fix_wrong_captured_features.py', description='fix.')
parser.add_argument('--input_training', type=str, required=True, help='input file with the training data, in pickle format.')
parser.add_argument('--fix_files', default=False, action='store_true', help='fix files')

FLAGS = parser.parse_args()

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)

start = su.print_and_time('Reading test data...',  past=start, file=sys.stderr)
if FLAGS.fix_files:
    su.read_pickled_data_to_fix_files(os.path.join(FLAGS.input_training, 'feats.validation'))
    su.read_pickled_data_to_fix_files(os.path.join(FLAGS.input_training, 'feats.train'))
    su.read_pickled_data_to_fix_files(os.path.join(FLAGS.input_training, 'feats.test'))


print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)