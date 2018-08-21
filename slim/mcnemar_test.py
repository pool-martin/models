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
from statsmodels.sandbox.stats.runs import mcnemar
from svm_layer import utils as su

os.environ['JOBLIB_TEMP_FOLDER'] = "/data/tmp"

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Predict the SVM decision.')
#parser.add_argument('--input_model', type=str, required=True, help='input trained model, in pickle format.')
#parser.add_argument('--input_test', type=str, required=True, help='input file with the test data, in pickle format.')
parser.add_argument('--output_predictions', type=str , help='input file with the test data, in isbi challenge format (default=stdout).')
parser.add_argument('--output_metrics', type=str, help='input file with the test data, in text format (default=stdout).')
parser.add_argument('--output_images', type=str, help='input file with the test data, in text format (default=stdout).')
parser.add_argument('--pool_by_id', type=str, default='none', help='pool answers of contiguous identical ids: none (default), avg, max, xtrm')
parser.add_argument('--compute_rolling_window', default=False, action='store_true', help='compute rolling_window')
parser.add_argument('--file_1', type=str, default='none', help='pool answers of contiguous identical ids: none (default), avg, max, xtrm')
parser.add_argument('--file_2', type=str, default='none', help='compute rolling_window')

FLAGS = parser.parse_args()

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)

print('\n Starting pandas stuff', end='', file=sys.stderr)
import pandas as pd

df = pd.read_csv(FLAGS.file_1, names=['Frame', 'prob_porn', 'score_porn'])
print('\n Could read the csv', end='', file=sys.stderr)
df = df.sort_values(by='Frame')
print('\n Sorted by frame', end='', file=sys.stderr)
def compare(row):
    if row['prob_porn'] >= 0.5:
        return 1
    else:
        return 0
print('\n Created prediction', end='', file=sys.stderr)

df['predictions'] = df.apply(compare, axis=1)

df2 = pd.read_csv(FLAGS.file_2, names=['Frame', 'prob_porn', 'score_porn'])
print('\n Could read the csv', end='', file=sys.stderr)
df2 = df2.sort_values(by='Frame')
print('\n Sorted by frame', end='', file=sys.stderr)
def compare(row):
    if row['prob_porn'] >= 0.5:
        return 1
    else:
        return 0
print('\n Created prediction', end='', file=sys.stderr)

df2['predictions'] = df2.apply(compare, axis=1)

print('\n Will run Mcnemar', end='', file=sys.stderr)
res = mcnemar(df['predictions'].values, df2['predictions'].values, exact=False, correction=True)

print(res, end='', file=sys.stderr)


print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)