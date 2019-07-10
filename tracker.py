import numpy as np
from filterpy.kalman import KalmanFilter
from metric import Metric
from sklearn.utils.linear_assignment_ import linear_assignment
from track import Track, KalmanTrack

class Tracker(object):

    def __init__(self, metric_str, metric_threshold=None, max_age=30, n_init=3):
        
        self.frame_count = 0
        self.tracks = []
        self.max_age = max_age
        self.n_init = n_init
        self.metric = Metric(metric_str)
        if metric_threshold is None:
            self.metric_threshold = self.metric.threshold
        else:
            self.metric_threshold = metric_threshold

    def associate_detections_to_tracks(self, detections, tracks):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        dist_matrix = self.metric.distance(detections, tracks)

        if self.metric_threshold < 0:
            dist_matrix = -dist_matrix
        matched_indices = linear_assignment(dist_matrix)

        unmatched_trackers = []
        unmatched_detections = []
        if len(detections) > len(tracks):
            unmatched_detections = {i for i in xrange(len(detections))}
            unmatched_detections.difference_update(set(matched_indices[:, 0]))
            unmatched_detections = list(unmatched_detections)

        else:
            unmatched_trackers = {i for i in xrange(len(tracks))}
            unmatched_trackers.difference_update(set(matched_indices[:, 1]))
            unmatched_trackers = list(unmatched_trackers)

        matches = []
        for m in matched_indices:
            if dist_matrix[m[0], m[1]] > self.metric_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def predict(self):
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if (np.any(np.isnan(pos))):
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
        trks = self.predict()
        ret = []
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.tracks):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanTrack(dets[i, :], self.n_init, self.max_age)
            self.tracks.append(trk)
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.get_state()[0]
            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.n_init or self.frame_count <= self.n_init)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.tracks.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))