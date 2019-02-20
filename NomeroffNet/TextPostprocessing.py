import sys, os
import asyncio

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextPostprocessings

async def textPostprocessingOneAsync(text, textPostprocessName):
    _textPostprocessName = textPostprocessName.replace("-", "_")
    if _textPostprocessName in dir(TextPostprocessings):
        TextPostprocessing = getattr(getattr(TextPostprocessings, _textPostprocessName), _textPostprocessName)
    else:
        TextPostprocessing = getattr(getattr(TextPostprocessings, "xx_xx"), "xx_xx")
    postprocessManager = TextPostprocessing()
    return postprocessManager.find(text)

async def textPostprocessingAsync(texts, textPostprocessNames):
    loop = asyncio.get_event_loop()
    promises = [loop.create_task(textPostprocessingOneAsync(text, textPostprocessName)) for text, textPostprocessName in zip(texts, textPostprocessNames)]
    if bool(promises):
        await asyncio.wait(promises)
    return [promise.result() for promise in promises]

def textPostprocessing(texts, textPostprocessNames):
    resTexts = []
    for text, textPostprocessName in zip(texts, textPostprocessNames):
        _textPostprocessName = textPostprocessName.replace("-", "_")
        if _textPostprocessName in dir(TextPostprocessings):
            TextPostprocessing = getattr(getattr(TextPostprocessings, _textPostprocessName), _textPostprocessName)
        else:
            TextPostprocessing = getattr(getattr(TextPostprocessings, "xx_xx"), "xx_xx")
        postprocessManager = TextPostprocessing()
        resTexts.append(postprocessManager.find(text))
    return resTexts
