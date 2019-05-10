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
from sklearn.model_selection import GridSearchCV
import pandas as pd

from svm_layer import utils as su

os.environ['JOBLIB_TEMP_FOLDER'] = "/data/tmp"

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Predict the SVM decision.')
parser.add_argument('--input_dir', type=str, required=True, help='input file with the training data, in pickle format.')
parser.add_argument('--max_iter_svm', type=int, default=1000, help='maximum number of interations for the linear svm.')
parser.add_argument('--max_iter_hyper', type=int, default=10, help='maximum number of interations for the hyperparameter search.')
parser.add_argument('--jobs', type=int, default=1, help='number of parallel jobs in the hyperparameter search.')
parser.add_argument('--svm_method', type=str, default='RBF', help='svm method to employ: RBF (default), LINEAR_DUAL, or LINEAR_PRIMAL.')
parser.add_argument('--output_model', type=str, required=True, help='output file to receive the model, in pickle format.')

FLAGS = parser.parse_args()

first = start = su.print_and_time('Setup...', file=sys.stderr)

valid_svm_methods = [ 'RBF', 'LINEAR_DUAL', 'LINEAR_PRIMAL' ]
if not FLAGS.svm_method in valid_svm_methods :
    print('--svm_method must be one of ', ', '.join(valid_svm_methods), file=sys.stderr)
    sys.exit(1)
SVM_LINEAR = FLAGS.svm_method == 'LINEAR_DUAL' or FLAGS.svm_method == 'LINEAR_PRIMAL'
SVM_DUAL = FLAGS.svm_method == 'LINEAR_DUAL'

SVM_MAX_ITER = FLAGS.max_iter_svm
HYPER_MAX_ITER = FLAGS.max_iter_hyper
HYPER_JOBS = FLAGS.jobs

first = start = su.print_and_time('Reading training data...', file=sys.stderr)
input_file = os.path.join(FLAGS.input_dir, 'joint.train_and_eval.predictions.pkl')
df = pd.read_pickle(input_file)
print('columns: ', list(df.columns.values))

num_samples = len(df.index)
min_gamma   = np.floor(np.log2(1.0/num_samples)) - 4.0
max_gamma   = min(3.0, min_gamma+32.0)
scale_gamma = max_gamma-min_gamma
print('\tSamples: ', num_samples, file=sys.stderr)
if not SVM_LINEAR :
    print('\tGamma: ', min_gamma, min_gamma+scale_gamma, file=sys.stderr)

# 'previous_labels', 'prob_porn', 'score_porn'
ids = df['Frame_saliency'].values
labels = df['previous_labels_saliency'].values.astype(np.int)
features = df[['prob_porn_saliency', 'score_porn_saliency', 'prob_porn_finetune', 'score_porn_finetune']].values

start = su.print_and_time('====================\nTraining porn  classifier...\n', past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
print('params :', classifier.get_params().keys(), file=sys.stderr)
#sys.exit(1)
classifier_m = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)

classifier_m.fit(features, (labels==1.).astype(np.int), groups=None)
print('Best params:', classifier_m.best_params_, file=sys.stderr)
print('...', classifier_m.best_params_, end='', file=sys.stderr)

start = su.print_and_time('====================\nWriting model...', past=start, file=sys.stderr)
model_dir = os.path.dirname(FLAGS.output_model)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_file = open(FLAGS.output_model, 'wb')
pickle.dump(classifier_m, model_file)
pickle.dump(FLAGS, model_file)
model_file.close()


print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)