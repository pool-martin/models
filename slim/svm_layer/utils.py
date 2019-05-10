from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import pickle
import sys

import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sk
import sklearn.decomposition 
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.preprocessing 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# class Preprocess_None(x) :     
#     def fit_transform(self, x) :
#         return x
#     def fit(self, x) :
#         return x

def print_and_time(*args, **kwargs) : 
    now = datetime.datetime.utcnow()
    past = kwargs.pop('past', None)
    if not past is None  :
        elapsed = (now-past).seconds
        elapsed_min = elapsed // 60
        elapsed_sec = elapsed % 60
        print(" elapsed: %d'%d''" % (elapsed_min, elapsed_sec), file=kwargs.get('file', sys.stdout))
    end=kwargs.pop('end', '')
    print(*args, end=end, **kwargs)
    return now

def removeDuplicates(ids, labels, features_in, feature_size, filename):
    print('Remove Duplicates', features_in.dtype, 'feat-shape: ', features_in.shape, labels.dtype, 'labels.shape: ', labels.shape)
    df = pd.DataFrame(data={'ids':ids, 'labels':labels, 'features':features_in.tolist()})
#    print(df)
    df = df.iloc[df.astype(str).drop_duplicates(subset='ids').index]
#    print(df)
    ids = df['ids'].values
    labels = df['labels'].values
    #df['features'] = df['features'].apply(lambda x: np.array(x))
    f_features = df['features'].values.tolist()

    features = np.empty([len(f_features), feature_size], dtype=np.float)
    for s in  range(len(f_features)):
        features[s] = f_features[s]


    print('Remove Duplicates - end', features.dtype, 'feat-shape: ', features.shape, labels.dtype, 'labels.shape: ', labels.shape)
    outfile = open(filename+".fix", 'wb')
    pickle.dump([len(f_features), feature_size], outfile)
    for image_id, label, feats in zip(ids, labels, features):
        pickle.dump([image_id, label, feats], outfile)
    outfile.close()

    return ids, labels, features

def read_pickled_data_to_fix_files(filename) :
    source = open(filename, 'rb')
#    source = open(filename+".fix", 'rb')
    sizes = pickle.load(source)
    num_samples = sizes[0] * 2
#    num_samples = sizes[0]
    feature_size = sizes[1]
    print("num_samples: %d feature_size %d" % (num_samples, feature_size))

    ids = []
    labels = np.empty([num_samples], dtype=np.float)
    print("labels and features 0 created")
    features = np.empty([num_samples, feature_size], dtype=np.float)
    for s in range(num_samples) :
        sample = pickle.load(source)
        ids.append(sample[0])
        labels[s] = sample[1]
        features[s] = sample[2]
    source.close()

    ids, labels, features = removeDuplicates(ids, labels, features, feature_size, filename)

#    print('result: ids', ids.shape, ' label: ', labels.shape, 'feat: ', features.shape)

    return ids, labels, features


def read_pickled_data(filename) :
    source = open(filename, 'rb')
    sizes = pickle.load(source)
    num_samples = sizes[0]
    feature_size = sizes[1]
    #Bellow we have a workaround to fix a defect when I created the files that stored the results of network
    if 'finetune' in filename:
         feature_size = feature_size -2
        num_samples = num_samples * 2

#    num_classes = sizes[2]
    print("num_samples: %d feature_size: %d " % (num_samples, feature_size))
	
    ids = []
    labels = np.empty([num_samples], dtype=np.float)
    print("labels and features 0 created")
    features = np.empty([num_samples, feature_size], dtype=np.float)
#    probs = np.empty([num_samples, num_classes], dtype=np.float)
    for s in range(num_samples) :
        sample = pickle.load(source)
        ids.append(sample[0])
        labels[s] = sample[1]
        features[s] = sample[2]
#        probs[s] = sample[3]
    source.close()
    return ids, labels, features

class exp2var() :
    def __init__(self, loc=0.0, scale=1.0) :
        self.dist = sp.stats.uniform(loc=loc, scale=scale)
        self.loc  = loc
        self.scale = scale
    def rvs(self, **kwargs) :
        u = self.dist.rvs(**kwargs)
        return 2.0**u

def new_classifier(randomForest=False, linear=False, dual=True, max_iter=10000, min_gamma=-24, scale_gamma=8, estimators_num=100) :
    if randomForest:
        parameters = {
            'n_estimators'      : [700, 800, 900, 1000, 1100, 1200],
#            'criterion'         : ['gini', 'entropy'],
#            'max_features'      : ['auto', 'log2', None],
#            'bootstrap'         : [False],
#            'min_samples_split ': [2, 0.01],
            'n_jobs': [4],
            'random_state'      : [0],
        }
        classifier = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    elif linear :
        parameters = {
            'dual'         : [ dual ],
            'C'            : np.logspace(-2, 10, 13), #exp2var(loc=-16.0, scale=32.0),
            'multi_class'  : [ 'ovr' ], 
            'random_state' : [ 0 ], 
            'max_iter'     : [ max_iter ],
        }
        classifier = sk.svm.LinearSVC()
    else :
        parameters = {
            'C'                       : exp2var(loc=-16.0, scale=32.0),
            'gamma'                   : exp2var(loc=min_gamma, scale=scale_gamma),
            'kernel'                  : [ 'rbf' ], 
            'decision_function_shape' : [ 'ovr' ], 
            'random_state'            : [ 0 ],
        }
        classifier = sk.svm.SVC()
    return classifier, parameters

def hyperoptimizer(classifier, parameters, scoring='roc_auc', max_iter=10, n_jobs=1, group=True) :
    return sk.model_selection.RandomizedSearchCV(classifier, parameters, 
        n_iter=max_iter, scoring=scoring, fit_params=None, n_jobs=n_jobs, iid=True, refit=True, 
        cv=3,
#        cv=sk.model_selection.GroupKFold(n_splits=3) if group else None, 
        verbose=2, random_state=0)