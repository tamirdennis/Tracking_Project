import numpy as np
from metric import Metric
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from track import Track, KalmanTrack, TrackStatus


class Tracker(object):

    def __init__(self, metric_str, metric_threshold=None, max_age=30, n_init=3, track_type=KalmanTrack):

        self.num_created_tracks = 0
        self.frame_count = 0
        self.tracks = []
        self.framed_matched_track_dets = []
        self.framed_unmatched_dets = []
        self.max_age = max_age
        self.n_init = n_init
        self.track_type = track_type
        self.metric = Metric(metric_str)
        if metric_threshold is None:
            self.metric_threshold = self.metric.threshold
        else:
            self.metric_threshold = metric_threshold

    def associate_detections_to_tracks(self, detections, tracks):
        """
        Assigns detections to tracked object - assuming both represented as bounding boxes in input!

        Updating self.framed_matched_track_dets list with the matched detections and tracks(as class) - list of tuples.
        Updating self.framed_unmatched_dets with the unmatched detections as bounding boxes
        """

        self.framed_matched_track_dets = []
        self.framed_unmatched_dets = []

        if len(tracks) == 0:
            self.framed_unmatched_dets = detections

        dist_matrix = self.metric.distance(detections, tracks)

        if self.metric_threshold < 0:
            dist_matrix = -dist_matrix
        row, col = linear_sum_assignment(dist_matrix)

        if len(detections) > len(tracks):
            self.framed_unmatched_dets = [det for d, det in enumerate(detections) if d not in row]

        for det, trk in zip(row, col):
            if dist_matrix[det, trk] > self.metric_threshold:
                self.framed_unmatched_dets.append(detections[det])
            else:
                self.framed_matched_track_dets.append((detections[det], self.tracks[trk]))

    def predict(self):
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.tracks), 2))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict()
            trk[:] = [pos[0], pos[1]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        return trks

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets[:, 0] = (dets[:, 0] + dets[:, 2]) / 2.0
        dets[:, 1] = (dets[:, 1] + dets[:, 3]) / 2.0
        dets = dets[:, :2]
        tracks_predicts = self.predict()
        ret = []
        self.associate_detections_to_tracks(dets, tracks_predicts)
        # update matched tracks with assigned detections

        for (det, trk) in self.framed_matched_track_dets:
            trk.update(det)
            if trk.is_tentative() and trk.hit_streak >= self.n_init:
                trk.status = TrackStatus.Confirmed

        # create and initialise new trackers for unmatched detections

        for det in self.framed_unmatched_dets:
            self.num_created_tracks += 1
            trk = self.track_type(self.num_created_tracks, det)
            self.tracks.append(trk)

        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.get_state()
            if trk.is_confirmed() or self.frame_count <= self.n_init:
                # ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                ret.append(trk)
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.tracks.pop(i)
        if len(ret) > 0:
            return ret
        return np.empty((0, 5))
