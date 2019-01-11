import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextDetectors


def TextDetector(textDetectorName):
    textDetectorName = textDetectorName.lower()
    if textDetectorName in dir(TextDetectors):
        textDetector = getattr(getattr(TextDetectors, textDetectorName), textDetectorName)
        return textDetector()
    else:
        raise Error(f"Text detector name '{textDetectorName}' not exists")