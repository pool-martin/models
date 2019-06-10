
from utils import opencv
import cv2, os
import skvideo.io
import gc
from joblib import Parallel, delayed
import pathlib
from PIL import Image
import pyflow
import numpy as np
import cv2


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
                                    type=str, required=False, default='/Exp/2kporn/splits_of/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/DL/2kporn/of_frames/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1_a')
    args = ap.parse_args()
    print(args)
    return args

def processOpticalFlow(args, video, frame, next_frame):
    im1 = np.array(Image.open(os.path.join(args.output_path, video, '{}.jpg'.format(frame))))
    im2 = np.array(Image.open(os.path.join(args.output_path, video, '{}.jpg'.format(next_frame))))
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(args.output_path, video, '{}.jpg'.format(frame)), rgb)

def file_exists(args, video, frame_identifier):
    file_path = os.path.join(args.output_path, video, '{}.jpg'.format(frame_identifier))
    if (os.path.isfile(file_path)):
        return True
    else:
        return False


def processVideoFrames(args, video, video_frames):
    global number_of_videos
    pairs = [[video_frames[i], video_frames[i + 1]] for i in range(len(video_frames) - 1)]
    for [frame, next_frame] in pairs:
        processOpticalFlow(args, video, frame, next_frame)
    number_of_videos += 1
    print('number_of_videos:', number_of_videos)




def extract(args, all_set):

    class_types = ['Porn', 'NonPorn']
    video_frames = {}

    for class_type in class_types:
        for i in range(1, 1001):
            video = 'v{}{}'.format(class_type, str(i).zfill(6))
            video_frames[video] = [frame for frame in all_set if video in frame]

    Parallel(n_jobs=10)(delayed(processVideoFrames)(args, video, video_frames[video]) for video in video_frames.keys())

    

def main():
    args = load_args()

    # /Exp/2kporn/splits_of/s1_a/2D/1_fps/opencv/
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