
from utils import opencv
import cv2, os
import skvideo.io
import gc
from joblib import Parallel, delayed
import pathlib

import argparse, os, time, random, math
from subprocess import call

number_of_videos = 0

def load_args():
    ap = argparse.ArgumentParser(description='Create Splits to be used in 2D or 3D CNN models.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/2kporn/')
    ap.add_argument('-sp', '--split-path',
                                    dest='split_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/Exp/2kporn/splits/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/2kporn/frames/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1_a')
    args = ap.parse_args()
    print(args)
    return args

def file_exists(args, video, frame_identifier):
    file_path = os.path.join(args.output_path, video, '{}.jpg'.format(frame_identifier))
    if (os.path.isfile(file_path)):
        return True
    else:
        return False


def extractVideoFrames(args, video, video_frames):
    global number_of_videos
    # frames_to_extract = [frame for frame in video_frames if not file_exists(args, video, frame)]
    frames_to_extract = []
    for identifier in video_frames:
        s_identifier = identifier.split('_')
        frames_path = os.path.join(args.split_path, args.split_number, '3D', '1_fps', 'opencv', 'w_1_l_1', video, '{}.txt'.format(identifier))
        content = []
        with open(frames_path, 'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        frames_to_extract.extend(['{}_{}_{}'.format(s_identifier[0], s_identifier[1], x) for x in content])

    frames_to_extract = [frame for frame in frames_to_extract if not file_exists(args, video, frame)]
    video_path = os.path.join(args.dataset_dir, 'videos', '{}.mp4'.format(video))
    print(video, ': ', frames_to_extract)
    opencv.extract_video_frames(video, video_path, args.output_path, frames_to_extract)
    number_of_videos += 1
    print('number_of_videos:', number_of_videos)


def extract(args, all_set):

    class_types = ['Porn', 'NonPorn']
    video_frames = {}

    for class_type in class_types:
        for i in range(1, 1001):
            video = 'v{}{}'.format(class_type, str(i).zfill(6))
            video_frames[video] = [frame for frame in all_set if video in frame]

    Parallel(n_jobs=10)(delayed(extractVideoFrames)(args, video, video_frames[video]) for video in video_frames.keys())

    

def main():
    args = load_args()

    splits_dir_path = os.path.join(args.split_path, args.split_number, '3D', '1_fps', 'opencv')

    network_training_set_path = os.path.join(splits_dir_path, 'network_training_set.txt')
    network_validation_set_path = os.path.join(splits_dir_path, 'network_validation_set.txt')
    test_set_path = os.path.join(splits_dir_path, 'test_set.txt')
    with open(network_training_set_path) as f:
        network_training_set = f.read().splitlines()
    with open(network_validation_set_path) as f:
        network_validation_set = f.read().splitlines()
    with open(test_set_path) as f:
        test_set = f.read().splitlines()

    all_set = network_training_set + network_validation_set + test_set
    extract(args, all_set)

if __name__ == '__main__':
    main()