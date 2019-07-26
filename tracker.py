import numpy as np
from metric import Metric
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from track import Track, TrackStatus, KalmanCentroidTrack, KalmanBBoxTrack, ParticleTrack


class Tracker(object):

    def __init__(self, metric_str, metric_threshold=None, max_age=30, n_init=3,
                 track_type=KalmanCentroidTrack, project=False, project_one=False):
        """

        :param metric_str(str): the metric string used as one of the metrics in Metric class. Must match to the track_type.
        :param metric_threshold(float): the threshold for the metric given - must be a match for the sign with the metric.
        default None and will take the default threshold from Metric class
        :param max_age(int): the maximum age of predicts for an unmatched track.
        :param n_init(int): the minimum number of matches for a track to exist.
        :param track_type(Track): the track type used. MUST match to the metric_str.
        :param project(bool): is projecting on image every update or not.
        :param project_one(bool):is projecting on image just one track at a time. project must be true for this to work.
        """
        self.one_track_projected = None
        self.project_one = project_one
        self.project = project
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
        Assigns detections to tracked object - assuming both represented as as the same representation the track type
        is expecting at update.

        Updating self.framed_matched_track_dets list with the matched detections and tracks(as class) - list of tuples.
        Updating self.framed_unmatched_dets with the unmatched detections.
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
        trks = np.zeros((len(self.tracks), self.track_type.track_dim))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict()
            trk[:] = [pos[i] for i in range(len(pos))]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)
        return trks

    def update(self, dets, image=None):
        """
        Params:
          return an array of tracks with type self.track_type that is still alive after updating.
          this function will project the relevant tracks if the relevant booleans are True.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets = self.track_type.precompile_detections(dets)
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

        if self.project:
            if self.project_one:
                if self.one_track_projected is not None:
                    if self.one_track_projected not in ret:
                        if len(ret) == 0:
                            self.one_track_projected = None
                        else:
                            self.one_track_projected = ret[0]

                if self.one_track_projected is None and len(ret) != 0:
                    self.one_track_projected = ret[0]

                if self.one_track_projected is not None:
                    self.one_track_projected.project_on_image(image)
            else:
                for track in ret:
                    track.project_on_image(image)
        return ret

