import cv2
import numpy as np
import os


def read_video(video_path, output_dir, ips):
    video_name = video_path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #print(num_frames//fps)

    n_trail = int(np.log10(num_frames//fps))+1
    #print(n_trail)


    #print(num_frames, fps)

    assert fps>=ips

    sel_frames = ((fps//ips)*np.arange(1,ips+1))%fps
    # print(sel_frames)

    s = 0
    last = 0
    for i in range(num_frames-1):
        img = cap.read()[1]
        if i%fps in sel_frames:
            if last >=i%fps:
                s+=1
            sec_num = (n_trail*'0'+str(s))[-n_trail:]
            #print(sec_num)
            cv2.imwrite(os.path.join(output_dir,video_name+ '_s'+sec_num+f'_f{i%fps}.jpg'), img)
            last = i%fps