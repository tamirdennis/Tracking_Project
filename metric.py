import numpy as np
from numba import jit
from scipy.spatial.distance import cdist, euclidean

@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def centroid_distance(det, tck):
    ctr_det = 0.5 * (det[:2] + det[2:4])
    ctr_trk = 0.5 * (tck[:2] + tck[2:4])
    return euclidean(ctr_det,ctr_trk)


class Metric(object):
    """
    Defining metric with different options for metrics defined in metrics_dict.
    The matching threshold for each metric is defined in threshold_dict.
    Threshold_dict values should be negative if large value is better metric for the metric
    and positive otherwise.
    """

    metrics_dict = {"iou": iou, "centroids": centroid_distance}
    threshold_dict = {"iou": -0.3, "centroids": 40}

    def __init__(self, metric_str):
        self.metric = Metric.metrics_dict[metric_str]
        self.threshold = Metric.threshold_dict[metric_str]

    def distance(self, detections, trackers):

        return cdist(detections, trackers, self.metric)





