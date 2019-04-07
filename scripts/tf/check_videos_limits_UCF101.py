#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  create_splits.py
#  
#  Copyright 2018 Joao Paulo Martin <joao.paulo.pmartin@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; 
#  either version 2 of the License, or (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
#  PURPOSE.  See the GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth 
#  Floor, Boston, MA 02110-1301, USA.
#  
#  

''' 
Check videos params agains real requests. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com)
'''


from utils import opencv
import cv2, os
import skvideo.io
import gc
from joblib import Parallel, delayed
import pathlib


from os.path import isfile, join


def check_frame_extraction(video_path, frame_no):
  cap = cv2.VideoCapture(video_path)
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
  (grabbed, frame) = cap.read()
  cap.release()
  return grabbed

def get_real_frame_count(video_path, frame_count):

  frame_no = frame_count
  result = False
  while(not result):
    frame_no = frame_no - 1
    result = check_frame_extraction(video_path, frame_no)

  return frame_no + 1

#vPorn000235.mp4 1 296.858 4.0107 event - porn - f
def get_localization_labels(filename):
    import operator
    i = 0
    content = {}

    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        values = line.split(' ')
        content[i] = [float(values[2]), float(values[3]), values[8][:1]]
        i += 1

    sorted_labels = sorted(content.items(), key=operator.itemgetter(0), reverse = True )
    #print "sorted_labels :%s" % sorted_labels
    return sorted_labels

def get_etf_duration(etf_path):
    labels = get_localization_labels(etf_path)
    etf_duration = (labels[0][1][0] + labels[0][1][1]) * 1000 
    return etf_duration

def get_video_real_duration(fps, last_valid_frame):
  duration = (last_valid_frame * 1000) / fps
  return duration

def get_real_values(video_path, fps, frame_count):
  real_frame_count = get_real_frame_count(video_path, frame_count)
  duration = get_video_real_duration(fps, real_frame_count)
  return real_frame_count, duration

def checkOpencvVideo(real_duration_path, video_path):
    frame_count, fps, height, width = opencv.get_video_params(video_path)
    real_frame_count, real_duration = get_real_values(video_path, fps, frame_count)

    video_name = video_path.split('.')[0][len('/DL/UCF-101/videos/'):]
    real_etf_path = os.path.join(real_duration_path, '{}.etf'.format(video_name))
    # cria o sub dir
    pathlib.Path(os.path.dirname(real_etf_path)).mkdir(parents=True, exist_ok=True)
    with open(real_etf_path, 'w') as f:
      f.write(str(int(real_frame_count)))
    
def listVideosInDir(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.avi' in file:
                files.append(os.path.join(r, file))
    return files
    
def mainOpenCVMultiThread():

    path = '/DL/UCF-101'
    videos_path = os.path.join(path, 'videos')

    real_duration_path = '/DL/UCF-101/frame_count'
    pathlib.Path(real_duration_path).mkdir(parents=True, exist_ok=True)

    videos = listVideosInDir(videos_path)

    Parallel(n_jobs=10)(delayed(checkOpencvVideo)(real_duration_path, path, video) for video in videos)


def mainOpenCV():

  path = '/DL/UCF-101'
  videos_path = os.path.join(path, 'videos')

  real_duration_path = '/DL/UCF-101/frame_count'
  pathlib.Path(real_duration_path).mkdir(parents=True, exist_ok=True)

  videos = listVideosInDir(videos_path)


  for video_path in videos:
    checkOpencvVideo(real_duration_path, video_path)

    
if __name__ == '__main__':
    #mainOpenCV()
    mainOpenCVMultiThread()