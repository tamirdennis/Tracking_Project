from enum import Enum
import numpy as np
from filterpy.kalman import KalmanFilter
from numpy.random import uniform, normal
from scipy.stats import norm


class TrackStatus(Enum):
    Confirmed = 2
    Deleted = 3
    Tentative = 1


class Track(object):

    def __init__(self, track_id, init_state, feature=None):
        self.curr_state = init_state
        self.id = track_id
        self.hits = 0

        self.features = []
        if feature is not None:
            self.features.append(feature)
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
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        return self.get_state()

    def update(self, detection):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.curr_state = detection

    def get_state(self):
        return self.curr_state


class KalmanTrack(Track):

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
        if (score == None):
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

        super(KalmanTrack, self).__init__(track_id, init_state)
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
        Updates the state vector with observed bbox.
        """
        super(KalmanTrack, self).update(centroid)
        self.kf.update(centroid)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        super(KalmanTrack, self).predict()
        # if ((self.kf.x[6] + self.kf.x[2]) <= 0):
        #     self.kf.x[6] *= 0.0
        self.kf.predict()

        return self.kf.x[:2]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:2]


class ParticleTrack(Track):

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
        # self.particles[:, 2], self.particles[:, 3] = uniform(-10, 10, size=self.N), uniform(-10, 10, size=self.N)
        self.particles[:, 2], self.particles[:, 3] = np.random.multivariate_normal([0, 0], self.R/1000., self.N).T

    def update(self, centroid):
        """
        Updates the state vector with observed bbox.
        """
        super(ParticleTrack, self).update(centroid)

        self.weights.fill(1.)
        dist = np.linalg.norm(self.particles[:, 0:2] - centroid, axis=1)
        # self.weights *= norm(dist, self.R).pdf(centroid)
        self.weights *= np.exp(-dist)
        self.weights /= sum(self.weights)  # normalize
        self.resample()
        # with the resample now, this only made it worse so I just resample every update:
        # if self.neff() < 0.95 * self.N:
        #     self.resample()

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        super(ParticleTrack, self).predict()

        self.particles[:, 0] += self.particles[:, 2]
        self.particles[:, 1] += self.particles[:, 3]

        # self.particles[:, 2] += normal(0, self.R[0][0]/1000., self.N)
        # self.particles[:, 3] += normal(0, self.R[0][0]/1000., self.N)

        # self.particles[:, 3] += uniform(-1, 1, size=self.N)

        return self.get_state()

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def resample(self):  # Multinomal resampling

        keep = self.weights > 0.05
        new_particles = np.zeros([self.N - np.sum(keep), 4])
        new_particles[:, 0], new_particles[:, 1] = np.random.multivariate_normal(self.get_state().ravel(), self.R, self.N - np.sum(keep)).T
        new_particles[:, 2], new_particles[:, 3] = np.random.multivariate_normal([0, 0], self.R/10, self.N - np.sum(keep)).T
        self.particles = np.r_[
            self.particles[keep, :],
            new_particles
        ]

    def get_state(self):
        x = [np.average(self.particles[:, 0], weights=self.weights, axis=0)]
        y = [np.average(self.particles[:, 1], weights=self.weights, axis=0)]
        return np.array([x, y])

