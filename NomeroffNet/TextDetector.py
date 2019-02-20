import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextDetectors

class TextDetector():
    def __init__(self, prisets):
        self.detectors_map = {}
        self.detectors = []

        i = 0
        for prisetName in prisets:
            priset = prisets[prisetName]
            for region in priset["for_regions"]:
                self.detectors_map[region] = i
            _label = prisetName
            if _label not in dir(TextDetectors):
                raise Exception(f"Text detector {_label} not in Text Detectors")
            TextPostprocessing = getattr(getattr(TextDetectors, _label), _label)
            detector = TextPostprocessing()
            detector.load(priset['model_path'])
            self.detectors.append(detector)
            i += 1

    def predict(self, zones, labels):
        predicted = {}

        orderAll = []
        resAll = []
        i = 0
        for zone, label in zip(zones, labels):
            if label in self.detectors_map.keys():
                detector = self.detectors_map[label]
                if detector not in predicted.keys():
                    predicted[detector] = {"zones": [], "order": []}
                predicted[detector]["zones"].append(zone)
                predicted[detector]["order"].append(i)
            else:
                resAll.append("")
                orderAll.append(i)
            i += 1

        for key in predicted.keys():
            resAll = resAll + self.detectors[int(key)].predict(predicted[key]["zones"])
            orderAll = orderAll + predicted[key]["order"]

        return [x for _, x in sorted(zip(orderAll,resAll), key=lambda pair: pair[0])]
