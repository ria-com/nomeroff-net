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


def translit_cyrillic_to_latin(cyrillic_str):
    cyrillic_str = cyrillic_str.upper()
    latin_str = ''
    for litter in cyrillic_str:
        if litter == "А":
            latin_str = f'{latin_str}A'
        elif litter == "В":
            latin_str = f'{latin_str}B'
        elif litter == "С":
            latin_str = f'{latin_str}C'
        elif litter == "Е":
            latin_str = f'{latin_str}E'
        elif litter == "Н":
            latin_str = f'{latin_str}H'
        elif litter == "І":
            latin_str = f'{latin_str}I'
        elif litter == "К":
            latin_str = f'{latin_str}K'
        elif litter == "М":
            latin_str = f'{latin_str}M'
        elif litter == "О":
            latin_str = f'{latin_str}O'
        elif litter == "Р":
            latin_str = f'{latin_str}P'
        elif litter == "Т":
            latin_str = f'{latin_str}T'
        elif litter == "Х":
            latin_str = f'{latin_str}X'
        else:
            latin_str = f'{latin_str}{litter}'
    return latin_str
