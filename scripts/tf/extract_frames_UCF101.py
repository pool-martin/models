
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
    ap = argparse.ArgumentParser(description='Create Splits to be used in 2D CNN models.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/UCF-101/')
    ap.add_argument('-sp', '--split-path',
                                    dest='split_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/UCF-101/splits/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/UCF-101/frames/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1')
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

    frames_to_extract = [frame for frame in video_frames if not file_exists(args, video, frame)]
    video_path = os.path.join(args.dataset_dir, 'videos', '{}.avi'.format(video))
    opencv.extract_video_frames(video, video_path, args.output_path, frames_to_extract)
    number_of_videos += 1
    print('number_of_videos:', number_of_videos)


def extract(args, all_set):

    #frame entry:
    # ApplyEyeMakeup/v_ApplyEyeMakeup_g16_c01.avi_1_0
    # video_name/path _ class _ frame
    all_set = [x.split(".avi_") for x in all_set]

    videos = set(all_set[0])

    video_frames = {}
    for video in videos:
        video_frames[video] = [x[1].split("_")[1] for x in all_set if video in x[0]]

    Parallel(n_jobs=10)(delayed(extractVideoFrames)(args, video, video_frames[video]) for video in video_frames.keys())

    

def main():
    args = load_args()

    splits_dir_path = os.path.join(args.split_path, args.split_number, '2D', '1_fps', 'opencv')

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