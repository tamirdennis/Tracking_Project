
# from __future__ import print_function

import os.path
import numpy as np
import time
import argparse

from track import KalmanCentroidTrack, ParticleTrack, KalmanBBoxTrack
from tracker import Tracker
import cv2
import scipy.stats as st


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
    # for metric centroid, use only centroid like tracks as track_type: ParticleTrack, KalmanCentroidTrack.
    # and for iou use only KalmanBBoxTrack
    metric = "centroids"
    # The track type to use:
    track_type = ParticleTrack
    # Change to True in order to see the track projection on only one track at a time:
    show_one_projection = False
    if (display):
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()

    if not os.path.exists('output'):
        os.makedirs('output')
    for _ in range(1):
        for seq in sequences:
            mot_tracker = Tracker(metric, max_age=5, track_type=track_type, n_init=3,
                                  project=display, project_one=show_one_projection, consider_features=True)  # create instance of the SORT tracker
            seq_dets = np.loadtxt('data/%s/det.txt' % (seq), delimiter=',')  # load detections
            with open('output/%s.txt' % (seq), 'w') as out_file:
                # print("Processing %s." % (seq))
                for frame in range(int(seq_dets[:, 0].max())):
                    frame += 1  # detection and frame numbers begin at 1
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:6]
                    dets[:, 2:4] += dets[:, 0:2]  # from x, y, w, h to x1, y1, x2, y2
                    total_frames += 1

                    if (display):
                        fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame)
                        im = cv2.imread(fn)

                    start_time = time.time()
                    if not display:
                        im = None
                    tracks = mot_tracker.update(dets, image=im)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    for trk in tracks:
                        trk_id_show = trk.id + 1
                        d = trk.get_state()
                        print('%d,%s' % (frame, trk.data_for_output_file()),
                              file=out_file)

                    if (display):
                        cv2.imshow('image', im)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                cv2.destroyAllWindows()

        print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
