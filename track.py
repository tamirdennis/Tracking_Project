from enum import Enum
import numpy as np
from filterpy.kalman import KalmanFilter


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
        super(KalmanTrack, self).update(bbox)
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        super(KalmanTrack, self).predict()
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()

        return self.convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)