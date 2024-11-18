import cv2
import argparse
import numpy as np
import os

from utils import read_video

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", 
                    type = str, 
                    help="choose the directory of videos you want to read, it must contain only videos")
parser.add_argument("--output_dir",
                    type = str,
                    default = None)
parser.add_argument("--ips",
                    type = int,
                    help = "number of sampled frame(s) per second",
                    default = 1)
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)



list_videos = os.listdir(args.dir_path)
for video_name in list_videos:
    video_path = os.path.join(args.dir_path, video_name)
    out = os.path.join(args.output_dir, video_name.split('.')[0])
    if not os.path.isdir(out):
        os.mkdir(out)
    read_video(video_path, out, args.ips)
