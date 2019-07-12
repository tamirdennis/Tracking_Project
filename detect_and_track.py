
from __future__ import print_function


import numpy as np
import time
import argparse
import cv2

from tracker import Tracker
from fl_detector import DetectorAPI


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

    mot_tracker = Tracker("iou", max_age=1)
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)
    args = parse_args()
    display = True
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

        dets = np.array([[boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2], scores[i]] for i in range(len(boxes)) if
                classes[i] == 1 and scores[i] > threshold]).reshape(-1,5)

        start_time = time.time()
        tracks = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        with open('output/%s.txt' % video_name, 'a+') as out_file:
            for d in tracks:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (total_frames, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                      file=out_file)
                if (display):
                    d = d.astype(np.int32)
                    rectangle_colors = (colours[d[4] % 32, :] * 255).astype(int)
                    cv2.rectangle(img, (d[0], d[1]), (d[2], d[3]), color=rectangle_colors, thickness=2)
                    cv2.putText(img, 'id: {}'.format(d[4]), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                rectangle_colors, 2)

            if (display):
                cv2.imshow('preview', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
