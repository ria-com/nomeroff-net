import sys
import os
import asyncio
from typing import List

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextPostprocessings


async def textPostprocessingOneAsync(text: str, text_postprocess_name: str) -> str:
    _textPostprocessName = text_postprocess_name.replace("-", "_")
    if _textPostprocessName in dir(TextPostprocessings):
        postprocess_manager = getattr(getattr(TextPostprocessings, _textPostprocessName), _textPostprocessName)
    else:
        postprocess_manager = getattr(getattr(TextPostprocessings, "xx_xx"), "xx_xx")
    return postprocess_manager.find(text)


async def textPostprocessingAsync(texts: List[str], text_postprocess_name: List[str]) -> List[str]:
    loop = asyncio.get_event_loop()
    promises = [loop.create_task(textPostprocessingOneAsync(text, textPostprocessName))
                for text, textPostprocessName in zip(texts, text_postprocess_name)]
    if bool(promises):
        await asyncio.wait(promises)
    return [promise.result() for promise in promises]


def textPostprocessing(texts: List[str], text_postprocess_name: List[str]) -> List[str]:
    res_texts = []
    for text, textPostprocessName in zip(texts, text_postprocess_name):
        _textPostprocessName = textPostprocessName.replace("-", "_")
        if _textPostprocessName in dir(TextPostprocessings):
            postprocess_manager = getattr(getattr(TextPostprocessings, _textPostprocessName), _textPostprocessName)
        else:
            postprocess_manager = getattr(getattr(TextPostprocessings, "xx_xx"), "xx_xx")
        res_texts.append(postprocess_manager.find(text))
    return res_texts
