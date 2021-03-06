import numpy as np
from metric import Metric
from scipy.optimize import linear_sum_assignment
from track import Track, TrackStatus, KalmanCentroidTrack, KalmanBBoxTrack, ParticleTrack
import cv2
from ReID_Model import get_siamese_model, INPUT_SHAPE, BEST_MODEL_PATH, process_image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import OrderedDict
import imgaug as ia
from skimage.transform import resize


class Tracker(object):

    reid_model = get_siamese_model(INPUT_SHAPE)
    reid_model.load_weights(BEST_MODEL_PATH)

    def __init__(self, metric_str, metric_threshold=None, max_age=30, n_init=3,
                 track_type=KalmanCentroidTrack, project=False, project_one=False,
                 consider_features=False, use_reid_model=False):
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
        self.use_reid_model = use_reid_model
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
    @staticmethod
    def _get_cropped_by_det(det, image):
        """
        :param det: a BBox detection as x1, y1, x2, y2.
        :param image: the entire frame.
        :return: cropped image by the bbox with extra 10 inches of height.
        """
        int_det = det.astype(int)
        return image[int_det[1]:int_det[3], int_det[0]:int_det[2]]
    @staticmethod
    def _get_reduced_crop_by_det(det, image):
        int_det = det.astype(int)
        width = int_det[2] - int_det[0]
        height = int_det[3] - int_det[1]
        return image[int_det[1] + int(height / 8):int_det[3] - int(height / 8),
                     int_det[0] + int(width / 8):int_det[2] - int(width / 8)]
    @staticmethod
    def _get_up_down_crops_by_det(det, image):
        """

        :param det: a BBox detection as x1, y1, x2, y2.
        :param image: the entire frame.
        :return: two cropped images by the bbox with extra 10 inches of height - upper crop and lower crop.
        """
        int_det = det.astype(int)
        h = det[3] - det[1]
        y1_up = int(det[1] + h / 2)
        crop_up = image[y1_up:int_det[3] + 5, int_det[0]:int_det[2]]
        crop_down = image[int_det[1] - 5:y1_up, int_det[0]:int_det[2]]
        return crop_up, crop_down

    @staticmethod
    def _compare_hists(hist1, hist2):
        """
        :param hist1: 3D histograms returned from Track.calc_hist function.
        :param hist2: 3D histograms returned from Track.calc_hist function.
        :return: the Hellinger distance between them.
        """
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

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
        # If we use features, then the next section will change the distance matrix using the tracks features.
        if self.consider_features:
            if image is None:
                raise ValueError("In order to consider the features, you have to pass the image frame each update.")
            fgmask = self.fgbg.apply(image)
            no_bg_image = cv2.bitwise_and(image, image, mask=fgmask)

            for det, trk in zip(row, col):
                if dist_matrix[det, trk] < self.metric_threshold:
                    if self.tracks[trk].time_since_features_ext > self.tracks[trk].features_ext_interval:
                        h = int(dets[det][3] - dets[det][1]) + 10
                        w = int(dets[det][2] - dets[det][0])
                        self.tracks[trk].extract_hist_features(no_bg_image, h, w)
                    if len(self.tracks[trk].features.keys()) == 0:
                        continue
                    col_dists = dist_matrix[:, trk]
                    if self.metric_threshold < 0:
                        r_add = - self.metric_threshold
                    else:
                        r_add = self.metric_threshold
                    det_indices = np.where(col_dists < dist_matrix[det, trk] + r_add)[0]
                    det_indices = np.unique(det_indices)
                    if len(det_indices) > 1:
                        trk_hist_up = self.tracks[trk].features["hist_up"]
                        trk_hist_down = self.tracks[trk].features["hist_down"]
                        crops = [Tracker._get_up_down_crops_by_det(dets[d_i], no_bg_image) for d_i in det_indices]
                        hists = [(Track.calc_hist(up_down_crops[0]), Track.calc_hist(up_down_crops[1]))
                                 for up_down_crops in crops]
                        diffs_hists = [max(Tracker._compare_hists(trk_hist_up, up_down_hist[0]),
                                           Tracker._compare_hists(trk_hist_down, up_down_hist[1]))
                                       for up_down_hist in hists]
                        for i, d_i in enumerate(det_indices):
                            if self.metric_threshold > 0:
                                dist_matrix[d_i, trk] *= diffs_hists[i]
                            else:
                                dist_matrix[d_i, trk] *= (1-diffs_hists[i])
            row, col = linear_sum_assignment(dist_matrix)
            if len(compiled_dets) > len(tracks_predicts):
                self.framed_unmatched_dets = [det for d, det in enumerate(compiled_dets) if d not in row]

            for det, trk in zip(row, col):
                if dist_matrix[det, trk] > self.metric_threshold:
                    self.framed_unmatched_dets.append(compiled_dets[det])
                else:
                    self.framed_matched_track_dets.append((compiled_dets[det], self.tracks[trk]))
        elif self.use_reid_model:
            if image is None:
                raise ValueError("In order to use reid model, you have to pass the image frame each update.")
            collected_pairs = OrderedDict()
            collected_matchings = {}
            for det, trk in zip(row, col):
                if dist_matrix[det, trk] < self.metric_threshold:
                    col_dists = dist_matrix[:, trk]
                    if len(col_dists) == 1:
                        continue
                    min_dist = np.min(col_dists)
                    next_min = np.min(col_dists[col_dists > min_dist])
                    have_feature = len(self.tracks[trk].features.keys()) != 0
                    # update the track feature "crop" only when its matching track is far from other detections:
                    if (self.tracks[trk].is_confirmed()
                            and self.tracks[trk].age % self.tracks[trk].features_ext_interval == 0) and (min_dist / next_min < 0.2
                            and min_dist == dist_matrix[det, trk]):
                        cropped = Tracker._get_cropped_by_det(dets[det], image.copy())
                        self.tracks[trk].features["crop"] = cropped
                    if not have_feature:
                        continue
                    col_dists = dist_matrix[:, trk]
                    det_indices = np.where(col_dists < self.metric_threshold)[0]
                    det_indices = np.unique(det_indices)
                    if len(det_indices) > 1:
                        # initializing a track for later with score and detection 0, for later use.
                        collected_matchings[trk] = (0, 0)
                        for d in det_indices:
                            d_crop = Tracker._get_reduced_crop_by_det(dets[d], image.copy())
                            collected_pairs[(d, trk)] = [d_crop, self.tracks[trk].features["crop"]]

            if len(collected_pairs) > 0:
                pairs = [np.zeros((len(collected_pairs), INPUT_SHAPE[0], INPUT_SHAPE[1], 3)) for _ in range(2)]
                for i, pair in enumerate(collected_pairs.values()):
                    pairs[0][i, :, :, :] = process_image(cv2.cvtColor(pair[0], cv2.COLOR_BGR2RGB))
                    pairs[1][i, :, :, :] = process_image(cv2.cvtColor(pair[1], cv2.COLOR_BGR2RGB))
                reid_model_similarities = Tracker.reid_model.predict(pairs).ravel()
                for i, pair_ind in enumerate(collected_pairs.keys()):
                    # for plotting the similarities: #
                    # pair = collected_pairs[pair_ind]
                    # title = "match score: {:.2f}".format(reid_model_similarities[i])
                    # fig = plt.figure(figsize=(20, 10))
                    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
                    # fig.suptitle(title)
                    # fig.add_subplot(1, 2, 1)
                    # plt.imshow(pair[0])
                    # fig.add_subplot(1, 2, 2)
                    # plt.imshow(pair[1])
                    # plt.show()

                    # if the current pair of track and detection has better similarity score then
                    # replace the collected matching with the current detection and its score:
                    if reid_model_similarities[i] > collected_matchings[pair_ind[1]][1]:
                        collected_matchings[pair_ind[1]] = (pair_ind[0], reid_model_similarities[i])

                # in order for the linear assignment to almost definitly pick the track and detection who matched best,
                # we put distance of 0 in the distance matrix
                for trk, value in collected_matchings.items():
                    dist_matrix[value[0], trk] = 0

            row, col = linear_sum_assignment(dist_matrix)
            if len(compiled_dets) > len(tracks_predicts):
                self.framed_unmatched_dets = [det for d, det in enumerate(compiled_dets) if d not in row]

            for det, trk in zip(row, col):
                if dist_matrix[det, trk] > self.metric_threshold:
                    self.framed_unmatched_dets.append(compiled_dets[det])
                else:
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