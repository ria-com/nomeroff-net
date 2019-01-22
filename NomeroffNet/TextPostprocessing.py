import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextPostprocessings


def textPostprocessing(text, textPostprocessName = "xx-xx", strong=False):
    _textPostprocessName = textPostprocessName.replace("-", "_")
    if _textPostprocessName in dir(TextPostprocessings):
        TextPostprocessing = getattr(getattr(TextPostprocessings, _textPostprocessName), _textPostprocessName)
    else:
        TextPostprocessing = getattr(getattr(TextPostprocessings, "xx_xx"), "xx_xx")
    postprocessManager = TextPostprocessing()
    return postprocessManager.find(text, strong)
