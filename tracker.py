import numpy as np
from metric import Metric
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from track import Track, TrackStatus, KalmanCentroidTrack, KalmanBBoxTrack, ParticleTrack
import cv2
from sklearn.preprocessing import MinMaxScaler


class Tracker(object):

    def __init__(self, metric_str, metric_threshold=None, max_age=30, n_init=3,
                 track_type=KalmanCentroidTrack, project=False, project_one=False, consider_features=False):
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
        self.consider_features = consider_features
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
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=8)
        if metric_threshold is None:
            self.metric_threshold = self.metric.threshold
        else:
            self.metric_threshold = metric_threshold

    def _get_cropped_by_det(self, det, image):
        int_det = det.astype(int)
        return image[int_det[1] - 10:int_det[3] + 5, int_det[0]:int_det[2]]

    def _associate_detections_to_tracks(self, detections, image):
        """
        Assigns detections to tracked object - assuming both represented as as the same representation the track type
        is expecting at update.

        Updating self.framed_matched_track_dets list with the matched detections and tracks(as class) - list of tuples.
        Updating self.framed_unmatched_dets with the unmatched detections.
        """
        dets = detections.copy()
        compiled_dets = self.track_type.precompile_detections(detections)
        self.framed_matched_track_dets = []
        self.framed_unmatched_dets = []
        tracks_predicts = self.predict()

        if len(tracks_predicts) == 0:
            self.framed_unmatched_dets = compiled_dets
            return
        dist_matrix = self.metric.distance(compiled_dets, tracks_predicts)
        if self.metric_threshold < 0:
            dist_matrix = -dist_matrix

        row, col = linear_sum_assignment(dist_matrix)

        if self.consider_features:
            if image is None:
                raise ValueError("In order to consider the features, you have to pass the image frame each update.")
            fgmask = self.fgbg.apply(image)
            no_bg_image = cv2.bitwise_and(image, image, mask=fgmask)

            for det, trk in zip(row, col):
                if dist_matrix[det, trk] < 2 * self.metric_threshold:
                    if len(self.tracks[trk].features.keys()) == 0:
                        continue
                    col_dists = dist_matrix[:, trk]
                    # dist_mul_ter
                    if self.metric_threshold < 0:
                        r_add = - self.metric_threshold

                    else:
                        r_add = self.metric_threshold
                    scaler = MinMaxScaler()
                    det_indices = np.where(col_dists < dist_matrix[det, trk] + r_add)[0]
                    det_indices = np.unique(det_indices)
                    if len(det_indices) > 1:
                        crops = [self._get_cropped_by_det(dets[d_i], no_bg_image) for d_i in det_indices]
                        hists = [cv2.calcHist([cropped], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
                                 for cropped in crops]
                        hists = [cv2.normalize(hist, hist).flatten() for hist in hists]
                        diffs_hists = [cv2.compareHist(self.tracks[trk].features["hist"], hist, cv2.HISTCMP_BHATTACHARYYA)
                                       for hist in hists]
                        # diffs_hists = scaler.fit_transform(np.array(diffs_hists).reshape(-1, 1))
                        for i, d_i in enumerate(det_indices):
                            if self.metric_threshold > 0:
                                dist_matrix[d_i, trk] *= diffs_hists[i]
                            else:
                                dist_matrix[d_i, trk] *= (1-diffs_hists[i])
                    # To visualize the score each time:
                    # if len(diffs_hists) > 1:
                    #
                    #     closest_crop = self._get_cropped_by_det(dets[det], no_bg_image)
                    #     closest_hist = cv2.calcHist([closest_crop], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
                    #     closest_hist = cv2.normalize(closest_hist, closest_hist).flatten()
                    #     diff_closest = cv2.compareHist(self.tracks[trk].features["hist"], closest_hist, cv2.HISTCMP_BHATTACHARYYA)
                        # for i, crop in enumerate(crops):
                        #     title = "track id: {}. closest crop score: {:.2f}. this score: {:.2f}".format(self.tracks[trk].id + 1, diff_closest, diffs_hists[i])
                        #     fig = plt.figure(figsize=(20, 10))
                        #     fig.subplots_adjust(hspace=0.1, wspace=0.1)
                        #     fig.suptitle(title)
                        #     fig.add_subplot(1, 2, 1)
                        #     plt.imshow(closest_crop)
                        #     fig.add_subplot(1, 2, 2)
                        #     plt.imshow(crop)
                        #     plt.show()
            row, col = linear_sum_assignment(dist_matrix)
            if len(compiled_dets) > len(tracks_predicts):
                self.framed_unmatched_dets = [det for d, det in enumerate(compiled_dets) if d not in row]

            for det, trk in zip(row, col):
                if dist_matrix[det, trk] > self.metric_threshold:
                    self.framed_unmatched_dets.append(compiled_dets[det])
                else:
                    if (self.tracks[trk].is_confirmed() and
                            self.tracks[trk].time_since_features_ext > self.tracks[trk].features_ext_interval
                            and self.tracks[trk].hit_streak >= 0) \
                            or self.tracks[trk].age == self.n_init:
                        cropped_img = self._get_cropped_by_det(dets[det], no_bg_image)
                        self.tracks[trk].extract_features(cropped_img)
                    self.framed_matched_track_dets.append((compiled_dets[det], self.tracks[trk]))
        else:
            if len(compiled_dets) > len(tracks_predicts):
                self.framed_unmatched_dets = [det for d, det in enumerate(compiled_dets) if d not in row]

            for det, trk in zip(row, col):
                if dist_matrix[det, trk] > self.metric_threshold:
                    self.framed_unmatched_dets.append(compiled_dets[det])
                else:
                    self.framed_matched_track_dets.append((compiled_dets[det], self.tracks[trk]))

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
        ret = []
        self._associate_detections_to_tracks(dets, image)
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


# class FeatureBasedTracker(Tracker):
#
#     def __init__(self, metric_str, metric_threshold=None, max_age=30, n_init=3,
#                  track_type=KalmanCentroidTrack, project=False, project_one=False):
#         """
#
#         :param metric_str(str): the metric string used as one of the metrics in Metric class. Must match to the track_type.
#         :param metric_threshold(float): the threshold for the metric given - must be a match for the sign with the metric.
#         default None and will take the default threshold from Metric class
#         :param max_age(int): the maximum age of predicts for an unmatched track.
#         :param n_init(int): the minimum number of matches for a track to exist.
#         :param track_type(Track): the track type used. MUST match to the metric_str.
#         :param project(bool): is projecting on image every update or not.
#         :param project_one(bool):is projecting on image just one track at a time. project must be true for this to work.
#         """
#         super(FeatureBasedTracker, self).__init__(metric_str, metric_threshold=metric_threshold, max_age=max_age, n_init=n_init,
#                  track_type=track_type, project=project, project_one=project_one)
#
#     def associate_detections_to_tracks(self, detections, tracks, image):
#         """
#         Assigns detections to tracked object - assuming both represented as as the same representation the track type
#         is expecting at update.
#
#         Updating self.framed_matched_track_dets list with the matched detections and tracks(as class) - list of tuples.
#         Updating self.framed_unmatched_dets with the unmatched detections.
#         """
#         dets = detections.copy()
#         compiled_dets = self.track_type.precompile_detections(detections)
#         self.framed_matched_track_dets = []
#         self.framed_unmatched_dets = []
#
#         if len(tracks) == 0:
#             self.framed_unmatched_dets = compiled_dets
#
#         dist_matrix = self.metric.distance(compiled_dets, tracks)
#
#         if self.metric_threshold < 0:
#             dist_matrix = -dist_matrix
#         row, col = linear_sum_assignment(dist_matrix)
#
#         if len(compiled_dets) > len(tracks):
#             self.framed_unmatched_dets = [det for d, det in enumerate(compiled_dets) if d not in row]
#
#         for det, trk in zip(row, col):
#             if dist_matrix[det, trk] > self.metric_threshold:
#                 self.framed_unmatched_dets.append(compiled_dets[det])
#             else:
#                 self.framed_matched_track_dets.append((compiled_dets[det], self.tracks[trk]))
#
#     def predict(self):
#         self.frame_count += 1
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.tracks), self.track_type.track_dim))
#         to_del = []
#         for t, trk in enumerate(trks):
#             pos = self.tracks[t].predict()
#             trk[:] = [pos[i] for i in range(len(pos))]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.tracks.pop(t)
#         return trks
#
#     def update(self, dets, image):
#         """
#         Params:
#           return an array of tracks with type self.track_type that is still alive after updating.
#           this function will project the relevant tracks if the relevant booleans are True.
#
#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         tracks_predicts = self.predict()
#         ret = []
#         self.associate_detections_to_tracks(dets, tracks_predicts)
#         # update matched tracks with assigned detections
#
#         for (det, trk) in self.framed_matched_track_dets:
#             trk.update(det)
#             if trk.is_tentative() and trk.hit_streak >= self.n_init:
#                 trk.status = TrackStatus.Confirmed
#
#         # create and initialise new trackers for unmatched detections
#
#         for det in self.framed_unmatched_dets:
#             self.num_created_tracks += 1
#             trk = self.track_type(self.num_created_tracks, det)
#             self.tracks.append(trk)
#
#         i = len(self.tracks)
#         for trk in reversed(self.tracks):
#             d = trk.get_state()
#             if trk.is_confirmed() or self.frame_count <= self.n_init:
#                 # ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
#                 ret.append(trk)
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 self.tracks.pop(i)
#
#         if self.project:
#             if self.project_one:
#                 if self.one_track_projected is not None:
#                     if self.one_track_projected not in ret:
#                         if len(ret) == 0:
#                             self.one_track_projected = None
#                         else:
#                             self.one_track_projected = ret[0]
#
#                 if self.one_track_projected is None and len(ret) != 0:
#                     self.one_track_projected = ret[0]
#
#                 if self.one_track_projected is not None:
#                     self.one_track_projected.project_on_image(image)
#             else:
#                 for track in ret:
#                     track.project_on_image(image)
#         return ret
