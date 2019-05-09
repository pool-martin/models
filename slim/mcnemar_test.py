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
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table

from svm_layer import utils as su

os.environ['JOBLIB_TEMP_FOLDER'] = "/data/tmp"

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Predict the SVM decision.')
parser.add_argument('--file_1', type=str, default='none', help='pool answers of contiguous identical ids: none (default), avg, max, xtrm')
parser.add_argument('--file_2', type=str, default='none', help='compute rolling_window')

FLAGS = parser.parse_args()

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)

print('\n Starting pandas stuff', end='', file=sys.stderr)
import pandas as pd

#    df = pd.read_csv(FLAGS.output_predictions+".k_test", names=['index','Frame', 'previous_labels', 'prob_porn', 'score_porn', 'predictions', 'prob_porn_2', 'videos', 'k_pred_b1', 'k_pred_b3', 'k_pred_b5', 'k_pred_t1', 'k_pred_t3', 'k_pred_t5'])
#    df = pd.read_csv(FLAGS.output_predictions+".k_test")
#    #print('\n Could read the csv', end='', file=sys.stderr)
#    df = df.sort_values(by='Frame')

df = pd.read_csv(FLAGS.file_1)
df = df.sort_values(by='Frame')
print('\n Sorted by frame', end='', file=sys.stderr)

df2 = pd.read_csv(FLAGS.file_2)
df2 = df2.sort_values(by='Frame')


def compare(row):
    return int(row['k_prob_g5'])

df['result_model_1'] = df.apply(compare, axis=1)
df2['result_model_2'] = df2.apply(compare, axis=1)

def compare2(row):
    return int(row['previous_labels'])
df['gt_labels'] = df.apply(compare2, axis=1)

print('\n Created final_results', end='', file=sys.stderr)

df.set_index('Frame').join(df2.set_index('Frame'), lsuffix='_model_1', rsuffix='_model_2')

print('\n joined', end='', file=sys.stderr)

print('\n Will run Mcnemar', end='', file=sys.stderr)

tb = mcnemar_table(y_target=df['gt_labels'].values, 
                   y_model1=df['result_model_1'].values, 
                   y_model2=df['result_model_2'].values)

print(tb)

chi2, p = mcnemar(ary=tb, exact=True)

print('chi-squared:', chi2)
print('p-value:', p)

print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)