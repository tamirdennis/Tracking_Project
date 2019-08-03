
# from __future__ import print_function


import numpy as np
import time
import argparse
import cv2

from track import KalmanCentroidTrack, KalmanBBoxTrack, ParticleTrack
from tracker import Tracker
from fl_detector import DetectorAPI
from metric import Metric


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    model_path_ssd = 'detectors_models/ssd_mobilenet_v1/frozen_inference_graph.pb'
    model_path_rcnn = 'detectors_models/faster_rcnn_inception_v2/frozen_inference_graph.pb'
    model_path = model_path_rcnn
    detector = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    '''
    Threshold should be 0.5 for ssd_mobilenet_v1
                        0.7 for faster_rcnn_inception_v2
    '''
    video_name = "aerial2.mp4"
    cap = cv2.VideoCapture('data/videos/{}'.format(video_name))

    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)
    args = parse_args()
    display = args.display
    # for metric centroid, use only centroid like tracks as track_type: ParticleTrack, KalmanCentroidTrack.
    # and for iou use only KalmanBBoxTrack
    metric = "centroids"
    # The track type to use:
    track_type = ParticleTrack
    # Change to True in order to see the track projection on only one track at a time:
    show_one_projection = False

    mot_tracker = Tracker(metric, max_age=5, track_type=track_type, n_init=3,
                          project=display, project_one=show_one_projection, consider_features=True)  # create instance of the SORT tracker

    open('output/%s.txt' % video_name, 'w+')

    while True:
        # read the video and resize
        r, img = cap.read()
        total_frames += 1
        if not r:
            break
        img = cv2.resize(img, (1280, 720))

        '''
        Pass the image frame to the detector, which in turn return
        the bounding boxes it detected for person in that frame, 
        together with their respective score and classification
        '''
        boxes, scores, classes, num = detector.processFrame(img)

        dets = np.array([[boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]] for i in range(len(boxes)) if
                classes[i] == 1 and scores[i] > threshold]).reshape(-1, 4)
        # dets[:, 2:4] -= dets[:, 0:2]
        start_time = time.time()
        tracks = mot_tracker.update(dets, image=img)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        with open('output/%s.txt' % video_name, 'a+') as out_file:
            for trk in tracks:
                trk_id_show = trk.id + 1
                d = trk.get_state()
                print('%d,%s' % (total_frames, trk.data_for_output_file()),
                      file=out_file)

            if (display):
                cv2.imshow('preview', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
