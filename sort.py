
from __future__ import print_function

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import time
import argparse

from tracker import Tracker
import cv2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof','ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17','ADL-Rundle-6','ADL-Rundle-8','Venice-2']

    args = parse_args()
    display = args.display
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if (display):
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()

    if not os.path.exists('output'):
        os.makedirs('output')

    for seq in sequences:
        mot_tracker = Tracker("centroids", max_age=1)  # create instance of the SORT tracker
        seq_dets = np.loadtxt('data/%s/det.txt' % (seq), delimiter=',')  # load detections
        with open('output/%s.txt' % (seq), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if (display):
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame)
                    im = cv2.imread(fn)

                start_time = time.time()
                tracks = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in tracks:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        rectangle_colors = (colours[d[4] % 32, :]*255).astype(int)
                        cv2.rectangle(im, (d[0], d[1]), (d[2], d[3]), color=rectangle_colors, thickness=2)
                        cv2.putText(im, 'id: {}'.format(d[4]),(d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    rectangle_colors, 2)

                if (display):
                    cv2.imshow('image', im)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    if (display):
        print("Note: to get real runtime results run without the option: --display")
