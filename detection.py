import numpy as np


class Detection(object):
    @staticmethod
    def parse_det(raw_det):
        return raw_det

    def __init__(self, raw_det, confidence=0, feature=""):
        self.parsed_det = self.parse_det(raw_det)
        self.confidence = confidence
        self.feature = feature


class BBoxDetection(Detection):

    def __init__(self, raw_det, confidence=0, feature=""):
        super(BBoxDetection, self).__init__(raw_det, confidence, feature)

    def to_tlbr(self):
        tlbr = [None]*4
        tlbr[0] = self.tlwh[0]
        tlbr[1] = self.tlwh[1]
        tlbr[2] = self.tlwh[0] + self.tlwh[2]
        tlbr[3] = self.tlwh[1] - self.tlwh[3]
        return tlbr

    def to_xyah(self):
        """
            Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
              [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
              the aspect ratio
            """
        bbox = self.to_tlbr()
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))


class CentroidDetection(Detection):

    @staticmethod
    def parse_det(raw_det):
        return 0.5 * (raw_det[:2] + raw_det[2:4])

    def __init__(self, raw_det, confidence=0, feature=""):
        super(BBoxDetection, self).__init__(raw_det, confidence, feature)

    def to_tlbr(self):
        tlbr = [None] * 4
        tlbr[0] = self.tlwh[0]
        tlbr[1] = self.tlwh[1]
        tlbr[2] = self.tlwh[0] + self.tlwh[2]
        tlbr[3] = self.tlwh[1] - self.tlwh[3]
        return tlbr

    def to_xyah(self):
        """
            Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
              [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
              the aspect ratio
            """
        bbox = self.to_tlbr()
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))