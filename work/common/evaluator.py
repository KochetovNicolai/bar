from model import SSDModel
from metrics import auc11
from converter import RectsInfo
import os


class Evaluator:
    def __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, img_root, conf_root, names, top=10, metric=auc11, verbose=100):

        print img_root
        total_qual = 0.0

        for i, name in enumerate(names):
            info = RectsInfo(name)
            info.load_rects(conf_root)
            img_name = os.path.join(img_root, os.path.splitext(info.file_name)[0]) + '.jpg'
            # print img_name
            pred = self.predictor.predict(img_name, top=top, threshold=None)
            qual = metric(info.rects, pred)
            total_qual += qual

            if verbose and (i + 1) % verbose == 0:
                print img_name
                print i + 1, 'avg', total_qual / (i + 1), 'last', qual

        return total_qual / len(names)
