from enum import Enum
import numpy as np
from filterpy.kalman import KalmanFilter
from numpy.random import uniform, normal
from scipy.stats import norm
import cv2
import os


class TrackStatus(Enum):
    Confirmed = 2
    Deleted = 3
    Tentative = 1


class Track(object):

    @staticmethod
    def precompile_detections(dets):
        """
        if the detections need to be precompiled before checking them on the tracks this function will be called.
        here we assume nothing on the dets format.
        :param dets: the detections in some format.
        :return: compiled detections
        """
        return dets

    colours = np.random.rand(32, 3)  # used only for display
    track_dim = 2

    def __init__(self, track_id, init_state, feature=None, features_ext_interval=5):
        self.curr_state = init_state
        self.id = track_id
        # os.mkdir("pictures_saved/{}".format(self.id))
        # self.count = 0
        self.hits = 0
        self.features_ext_interval = features_ext_interval
        self.time_since_features_ext = features_ext_interval + 1
        self.features = {}
        if feature is not None:
            self.features["init_feature"] = feature
        self.age = 0
        self.time_since_update = 0
        self.status = TrackStatus.Tentative
        self.hit_streak = 0

    def is_confirmed(self):
        return self.status == TrackStatus.Confirmed

    def is_deleted(self):
        return self.status == TrackStatus.Deleted

    def is_tentative(self):
        return self.status == TrackStatus.Tentative

    def mark_missed(self):
        pass

    def predict(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.time_since_features_ext += 1
        self.age += 1
        return self.get_state()

    def update(self, detection):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.curr_state = detection

    def get_state(self):
        return self.curr_state

    def extract_features(self, cropped_img):
        # self.count += 1
        # kernel = np.ones((2, 2), np.uint8)
        # self.features = cv2.morphologyEx(cropped_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        # cv2.imwrite("pictures_saved/{}/{}.png".format(self.id, self.count), cropped_img)
        # self.count += 1
        # cv2.imshow("cropped", cropped_img)
        # cv2.waitKey(0)
        hist = cv2.calcHist([cropped_img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        self.features["hist"] = hist
        self.time_since_features_ext = 0

    def data_for_output_file(self):
        pass

    def project_on_image(self, image):
        pass

    def _cent_data_to_output(self):
        d = self.get_state()
        return '%d,%.2f,%.2f' % (self.id + 1, d[0], d[1])

    def _bbox_data_to_output(self):
        d = self.get_state()
        return '%d,%.2f,%.2f,%.2f,%.2f' % (self.id + 1, d[0], d[1], d[2] - d[0], d[3] - d[1])

    def _project_centroid_on_image(self, image):
        trk_id_show = self.id + 1
        d = self.get_state()
        d = d.astype(np.int32)
        centroid_colors = (Track.colours[trk_id_show % 32, :] * 255).astype(float)
        cv2.circle(image, (d[0], d[1]), 10, color=centroid_colors, thickness=2)
        cv2.putText(image, 'id: {}'.format(trk_id_show), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    centroid_colors, 2)

    def _project_bbox_on_image(self, image):
        trk_id_show = self.id + 1
        d = self.get_state()
        d = d.astype(np.int32)
        rectangle_colors = (Track.colours[trk_id_show % 32, :] * 255).astype(float)
        cv2.rectangle(image, (d[0], d[1]), (d[2], d[3]), color=rectangle_colors, thickness=2)
        cv2.putText(image, 'id: {}'.format(trk_id_show), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    rectangle_colors, 2)


class KalmanCentroidTrack(Track):

    """
    This class represents the internel state of individual tracked objects observed as centroid.
    """
    track_dim = 2

    @staticmethod
    def precompile_detections(dets):
        """
        if the detections need to be precompiled before checking them on the tracks this function will be called.
        here we assume the dets format is [x1,y1,w,h] and will convert them to centroids
        :param dets: the detections in some format.
        :return: compiled detections
        """
        dets[:, 0] = (dets[:, 0] + dets[:, 2]) / 2.0
        dets[:, 1] = (dets[:, 1] + dets[:, 3]) / 2.0
        dets = dets[:, :2]
        return dets

    def __init__(self, track_id, init_state):
        """
        Initialises a tracker using initial centroid.
        """
        # define constant velocity model

        super(KalmanCentroidTrack, self).__init__(track_id, init_state)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.init_kalman_filter(init_state)

    def init_kalman_filter(self, init_state):

        self.kf.F = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]])

        self.kf.P[2:, 2:] *= 10.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[2:, 2:] *= 0.01

        self.kf.x[:2] = init_state.reshape(2, 1)

    def update(self, centroid):
        """
        Updates the state vector with observed centroid.
        """
        super(KalmanCentroidTrack, self).update(centroid)
        self.kf.update(centroid)

    def predict(self):
        """
        Advances the state vector and returns the predicted centroid estimate.
        """
        super(KalmanCentroidTrack, self).predict()
        # if ((self.kf.x[6] + self.kf.x[2]) <= 0):
        #     self.kf.x[6] *= 0.0
        self.kf.predict()

        return self.kf.x[:2]

    def get_state(self):
        """
        Returns the current centroid estimate.
        """
        return self.kf.x[:2]

    def project_on_image(self, image):
        self._project_centroid_on_image(image)

    def data_for_output_file(self):
        return self._cent_data_to_output()


class KalmanBBoxTrack(Track):

    track_dim = 4
    @staticmethod
    def precompile_detections(dets):
        """
        if the detections need to be precompiled before checking them on the tracks this function will be called.
        :param dets: the detections in some format.
        :return: compiled detections
        """
        return dets

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
          [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
          the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
          [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score == None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """

    def __init__(self, track_id, init_state):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model

        super(KalmanBBoxTrack, self).__init__(track_id, init_state)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.init_kalman_filter(init_state)

    def init_kalman_filter(self, init_state):
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(init_state)

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        super(KalmanBBoxTrack, self).update(bbox)
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        super(KalmanBBoxTrack, self).predict()
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

        return self.convert_x_to_bbox(self.kf.x)[0]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)[0]

    def project_on_image(self, image):
        self._project_bbox_on_image(image)

    def data_for_output_file(self):
        return self._bbox_data_to_output()


class ParticleTrack(Track):
    track_dim = 2
    @staticmethod
    def precompile_detections(dets):
        """
        if the detections need to be precompiled before checking them on the tracks this function will be called.
        here we assume the dets format is [x1,y1,w,h] and will convert them to centroids
        :param dets: the detections in some format.
        :return: compiled detections
        """
        dets[:, 0] = (dets[:, 0] + dets[:, 2]) / 2.0
        dets[:, 1] = (dets[:, 1] + dets[:, 3]) / 2.0
        dets = dets[:, :2]
        return dets

    def __init__(self, track_id, init_state, N=10000, x_or_y_error=100):
        super(ParticleTrack, self).__init__(track_id, init_state)

        self.N = N

        self.R = np.array([[x_or_y_error, 0], [0, x_or_y_error]])

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        self.weights.fill(1. / N)
        self.state = init_state

        self.particles = np.empty((N, 4))  # x, y, x_dot, y_dot
        self.particles[:, 0], self.particles[:, 1] = np.random.multivariate_normal(init_state, self.R, self.N).T
        self.particles[:, 2], self.particles[:, 3] = np.random.multivariate_normal([0, 0], self.R/1000., self.N).T

    def update(self, centroid):
        """
        Updates the state vector with observed centroid.
        """
        super(ParticleTrack, self).update(centroid)

        self.weights.fill(1.)
        dist = np.linalg.norm(self.particles[:, 0:2] - centroid, axis=1)
        # Deviding the weights by their exponent of their distance:
        self.weights *= np.exp(-dist)
        self.weights /= sum(self.weights)  # normalize
        self.resample()
        # with the resample we got, the next commented lines only made it worse so I just resample every update:
        # (this is because every resample I only resampled part of the particles.
        # if self.neff() < 0.95 * self.N:
        #     self.resample()

    def predict(self):
        """
        Advances the state vector and returns the predicted centroid estimate.
        """
        super(ParticleTrack, self).predict()
        # updated according to velocities
        self.particles[:, 0] += self.particles[:, 2]
        self.particles[:, 1] += self.particles[:, 3]
        # in my case adding noise to the velocitied didnt work well, but might be useful:
        # self.particles[:, 2] += normal(0, self.R[0][0]/1000., self.N)
        # self.particles[:, 3] += normal(0, self.R[0][0]/1000., self.N)

        return self.get_state()

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def resample(self):  # Multinomal resampling

        keep = self.weights > 0.15
        new_particles = np.zeros([self.N - np.sum(keep), 4])
        new_particles[:, 0], new_particles[:, 1] = np.random.multivariate_normal(self.get_state().ravel(), self.R, self.N - np.sum(keep)).T
        new_particles[:, 2], new_particles[:, 3] = np.random.multivariate_normal([0, 0], self.R/100, self.N - np.sum(keep)).T
        self.particles = np.r_[
            self.particles[keep, :],
            new_particles
        ]

    def get_state(self):
        x = [np.average(self.particles[:, 0], weights=self.weights, axis=0)]
        y = [np.average(self.particles[:, 1], weights=self.weights, axis=0)]
        return np.array([x, y])

    def project_on_image(self, image):
        # for particle in self.particles[:, :2]:
        #     particle = particle.astype(int)
        #     cv2.circle(image, (particle[0], particle[1]), 1, color=[0., 0., 255.])
        self._project_centroid_on_image(image)

    def data_for_output_file(self):
        return self._cent_data_to_output()

