import argparse
import math
import sys
import os

def read_content(file, line=0):
	with open(file, 'r') as f:
		content = f.readlines()[line]
	return content

def check_wrong_duration(video_name, etf_file, etf_fps, etf_frames, FLAGS):
	etf_last_line = read_content(etf_file, -1)
	fps = float(read_content(etf_fps))
	number_frames = float(read_content(etf_frames))
	
	line_split = etf_last_line.split(' ')
	etf_duration = float(line_split[2]) + float(line_split[3])
	
	video_duration = number_frames/fps
	
	difference = abs(video_duration - etf_duration)
	if(difference > FLAGS.allowed_difference):
		print(video_name, 'difference: {0:0>3.3} video_duration: {1}:{2} etf_duration: {3}:{4} number_frames: {5} fps: {6} line: {7}'.format(difference, int(video_duration/60), int(video_duration%60), int(etf_duration/60), int(etf_duration%60), number_frames, fps, etf_last_line))


def main():
	parser = argparse.ArgumentParser(prog='check_etf_length.py', description='verify if etfs match lenght in seconds with video time.')
	parser.add_argument('--dataset_dir', type=str , default='/home/jp/DL/2kporn/', help='dir where is the videos, etfs and stuff of dataset.')
	parser.add_argument('--allowed_difference', type=float , default=1.0, help='Allowed difference in seconds between etf and video.')
	FLAGS = parser.parse_args()

	etf_dir = os.path.join(FLAGS.dataset_dir, 'etf')
	etf_fps_dir = os.path.join(FLAGS.dataset_dir, 'video_fps')
	etf_frames_dir = os.path.join(FLAGS.dataset_dir, 'number_of_frames_video')
	
	for class_type in ['vPorn', 'vNonPorn']:
		for i in range(1, 1001):
			video_name = '{0}{1:0>6}'.format(class_type, i)
			etf_file_name = '{}{}'.format(video_name, '.etf')
			etf_file   = os.path.join(etf_dir, etf_file_name)
			etf_fps    = os.path.join(etf_fps_dir, etf_file_name)
			etf_frames = os.path.join(etf_frames_dir, etf_file_name)
			check_wrong_duration(video_name, etf_file, etf_fps, etf_frames, FLAGS)

if __name__ == '__main__':
	main()
