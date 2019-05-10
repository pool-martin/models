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
import pandas as pd

from svm_layer import utils as su

os.environ['JOBLIB_TEMP_FOLDER'] = "~/tmp"

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Predict the SVM decision.')
parser.add_argument('--input_model', type=str, required=True, help='input trained model, in pickle format.')
parser.add_argument('--output_predictions', type=str , help='input file with the test data, in isbi challenge format (default=stdout).')
parser.add_argument('--input_split', type=str, required=True, help='input split to extract.')

FLAGS = parser.parse_args()

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)
input_model = os.path.join('/Exp/2kporn/art/inception_v4', FLAGS.input_split, 'saliency/svm.models/svm.model')
model_file = open(input_model, 'rb')
preprocessor = pickle.load(model_file)
classifier_m = pickle.load(model_file)
model_file.close()


first = start = su.print_and_time('Reading train data...', file=sys.stderr)

input_training = os.path.join('/Exp/2kporn/art/inception_v4', FLAGS.input_split, 'saliency/svm.features')
ids_train, labels_train, features_train = su.read_pickled_data(os.path.join(input_training, 'feats.train'))
ids_val, labels_val, features_val = su.read_pickled_data(os.path.join(input_training, 'feats.validation'))
ids_test, labels_test, features_test = su.read_pickled_data(os.path.join(input_training, 'feats.test'))

start = su.print_and_time('', past=start, file=sys.stderr)
image_ids = np.append(ids_train, ids_val)
labels = np.append(labels_train, labels_val)
features = np.append(features_train, features_val, axis=0)

start = su.print_and_time('Preprocessing test data...', past=start, file=sys.stderr)
features = preprocessor.transform(features)
features_test = preprocessor.transform(features_test)

# "Probabilities" should come between quotes here
# Only if the scores are true logits the probabilities will be consistent
def probability_from_logits(logits) :
    odds = np.exp(logits)
    return odds/(odds+1.0)
def logits_from_probability(prob) :
    with np.errstate(divide='ignore') :
      odds = prob/(1.0-prob)
      return np.log(odds)
def extreme_probability(prob) :
  return prob[np.argmax(np.abs(logits_from_probability(prob)))]

start = su.print_and_time('Predicting train and validation data...\n', past=start, file=sys.stderr)
confidence_scores_train = classifier_m.decision_function(features)
confidence_scores_test = classifier_m.decision_function(features_test)

predictions_train = probability_from_logits(confidence_scores_train)
predictions_test = probability_from_logits(confidence_scores_test)

if not os.path.exists(FLAGS.output_predictions):
    os.makedirs(FLAGS.output_predictions)

start = su.print_and_time('Save to file ...\n', past=start, file=sys.stderr)

outfile = open(os.path.join(FLAGS.output_predictions, 'saliency.train_and_val.predictions'), 'w') if FLAGS.output_predictions else sys.stdout
for i in range(len(image_ids)) :
  print(image_ids[i].decode("utf-8"), labels[i], predictions_train[i], confidence_scores_train[i], sep=',', file=outfile)
outfile.close()

outfile = open(os.path.join(FLAGS.output_predictions, 'saliency.test.predictions'), 'w') if FLAGS.output_predictions else sys.stdout
for i in range(len(image_ids)) :
  print(ids_test[i].decode("utf-8"), labels_test[i], predictions_test[i], confidence_scores_test[i], sep=',', file=outfile)
outfile.close()

##########################################
start = su.print_and_time('\n\n\n##################\nExtract for finetune experiment ...\n', past=start, file=sys.stderr)

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)
input_model = os.path.join('/Exp/2kporn/art/inception_v4', FLAGS.input_split, 'finetune/svm.models/svm.model')
model_file = open(input_model, 'rb')
preprocessor = pickle.load(model_file)
classifier_m = pickle.load(model_file)
model_file.close()

input_training = os.path.join('/Exp/2kporn/art/inception_v4', FLAGS.input_split, 'finetune/svm.features')
ids_train, labels_train, features_train = su.read_pickled_data(os.path.join(input_training, 'feats.train'))
ids_val, labels_val, features_val = su.read_pickled_data(os.path.join(input_training, 'feats.validation'))
ids_test, labels_test, features_test = su.read_pickled_data(os.path.join(input_training, 'feats.test'))

start = su.print_and_time('', past=start, file=sys.stderr)
image_ids = np.append(ids_train, ids_val)
labels = np.append(labels_train, labels_val)
features = np.append(features_train, features_val, axis=0)

start = su.print_and_time('Preprocessing training data...', past=start, file=sys.stderr)
features = preprocessor.transform(features)
features_test = preprocessor.transform(features_test)


start = su.print_and_time('Predicting train and validation data...\n', past=start, file=sys.stderr)
confidence_scores_train = classifier_m.decision_function(features)
confidence_scores_test = classifier_m.decision_function(features_test)

predictions_train = probability_from_logits(confidence_scores_train)
predictions_test = probability_from_logits(confidence_scores_test)

output_dir = os.path.dirname(FLAGS.output_predictions)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start = su.print_and_time('Save to file ...\n', past=start, file=sys.stderr)

outfile = open(os.path.join(FLAGS.output_predictions, 'finetune.train_and_val.predictions'), 'w') if FLAGS.output_predictions else sys.stdout
for i in range(len(image_ids)) :
  print(image_ids[i].decode("utf-8"), labels[i], predictions_train[i], confidence_scores_train[i], sep=',', file=outfile)
outfile.close()

outfile = open(os.path.join(FLAGS.output_predictions, 'finetune.test.predictions'), 'w') if FLAGS.output_predictions else sys.stdout
for i in range(len(image_ids)) :
  print(ids_test[i].decode("utf-8"), labels_test[i], predictions_test[i], confidence_scores_test[i], sep=',', file=outfile)
outfile.close()

#############################################

df = pd.read_csv(os.path.join(FLAGS.output_predictions, 'saliency.train_and_val.predictions'), names=['Frame', 'previous_labels', 'prob_porn', 'score_porn'])
df = df.sort_values(by='Frame')
print('\n Sorted by frame', end='', file=sys.stderr)

df2 = pd.read_csv(os.path.join(FLAGS.output_predictions, 'finetune.train_and_val.predictions'), names=['Frame', 'previous_labels', 'prob_porn', 'score_porn'])
df2 = df2.sort_values(by='Frame')


dfjoined = df.set_index('Frame').join(df2.set_index('Frame'), lsuffix='_saliency', rsuffix='_finetune')
dfjoined.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

print('\n joined', end='', file=sys.stderr)

dfjoined.to_pickle(os.path.join(FLAGS.output_predictions, 'joint.train_and_eval.predictions.pkl'))

#############################################
print('=================\n process test files', end='', file=sys.stderr)

df = pd.read_csv(os.path.join(FLAGS.output_predictions, 'saliency.test.predictions'), names=['Frame', 'previous_labels', 'prob_porn', 'score_porn'])
df = df.sort_values(by='Frame')
print('\n Sorted by frame', end='', file=sys.stderr)

df2 = pd.read_csv(os.path.join(FLAGS.output_predictions, 'finetune.test.predictions'), names=['Frame', 'previous_labels', 'prob_porn', 'score_porn'])
df2 = df2.sort_values(by='Frame')


dfjoined = df.set_index('Frame').join(df2.set_index('Frame'), lsuffix='_saliency', rsuffix='_finetune')
dfjoined.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

print('\n joined', end='', file=sys.stderr)

dfjoined.to_pickle(os.path.join(FLAGS.output_predictions, 'joint.test.predictions.pkl'))

print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)