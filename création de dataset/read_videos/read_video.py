import cv2
import argparse
import numpy as np
import os

from utils import read_video

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", 
                    type = str, 
                    help="choose the video you want to read")
parser.add_argument("--output_dir",
                    type = str,
                    default = None)
parser.add_argument("--ips",
                    type = int,
                    help = "number of sampled frame(s) per second",
                    default = 1)
args = parser.parse_args()

default_path = '/home/msouda/Workspace/Video_manipulation/Read_videos/' #A CHANGER
vid = args.video_path.split('/')[-1].split('.')[0]


if args.output_dir is None:
    output_dir = os.path.join(default_path, vid)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
else:
    output_dir = os.path.join(args.output_dir, vid)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    

read_video(args.video_path, output_dir, args.ips)