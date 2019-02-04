from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import pickle
import sys
import os
import pandas as pd
from subprocess import call

def main():
    parser = argparse.ArgumentParser(prog='collect_errors.py', description='create etfs based on results.')
    parser.add_argument('--output_predictions', type=str , help='input file with the test data, in isbi challenge format (default=stdout).')
    parser.add_argument('--output_path', type=str, default='/Exp/2kporn/art/inception_v4/s1_a/finetune/error_examples' , help='folder to save the etf files.')
    parser.add_argument('--fold_to_process', type=str, default='s1_a', help='Wich fold should be processed for example s1, s2, ...')
    parser.add_argument('--column', type=str, default='k_prob_g5', help='Wich column to extract results, k_prob_t5, k_prob_t3, k_pred_t5, ...')
    FLAGS = parser.parse_args()

    df = pd.read_csv(FLAGS.output_predictions+".k_test")
    df = df.sort_values(by='Frame')

    for row in df.itertuples(index=True, name='Pandas'):
        truth_label = getattr(row, "previous_labels")
        identifier = getattr(row, "Frame")
        column = getattr(row, FLAGS.column)[0]
        video = getattr(row, 'videos')
        if int(truth_label) == 0:
            error_type = 'fn'
        else:
            error_type = 'fp'
        if int(truth_label) != int(column):
            print(identifier, truth_label, column, error_type)
            origin_path = os.path.join('/DL/2kporn/frames', video, '{}.jpg'.format(identifier))
            dest_path = os.path.join(FLAGS.output_path, FLAGS.fold_to_process, error_type, video)
            if not os.path.isdir(dest_path):
                os.makedirs(dest_path)

            command = "cp {} {}".format(origin_path, dest_path)
            print('\n', command)
            call(command, shell=True)


if __name__ == '__main__':
	main()
