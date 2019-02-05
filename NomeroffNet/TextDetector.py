import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextDetectors


def TextDetector(textDetectorEngine = "TESSERACT"):
    textDetectorEngine = textDetectorEngine.lower()
    if textDetectorEngine in dir(TextDetectors):
        textDetector = getattr(getattr(TextDetectors, textDetectorEngine), textDetectorEngine)
        return textDetector()
    else:
        raise Exception(f"Text detector name '{textDetectorEngine}' not exists")