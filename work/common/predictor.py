from model import SSDModel
from converter import Converter
import numpy as np
from PIL import Image


from metrics import RectWithConf


class Predictor:
    def __init__(self, ssd_model, ssd_converter):
        self.ssd_model = ssd_model
        self.ssd_converter = ssd_converter

    def predict(self, path, top=10, threshold=None):
        img = np.asarray(Image.open(path))
        tensor = self.ssd_model.model.predict(img.reshape([1] + list(img.shape)))
        res = self.ssd_converter.restore_rects_batch(tensor, self.ssd_model, top=top, threshold=threshold)[0]
        #print 42
        #print res
        confs, rects = res
        #print confs
        #print rects
        for rect in rects:
            rect.stretch(300, 300)
        return sorted([RectWithConf(r, c) for r, c in zip(rects, confs)], key=lambda x: x.conf, reverse=True)
