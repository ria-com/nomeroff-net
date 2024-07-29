def postprocess_multiline_text(text: str, count_line: int) -> str:
    if count_line != 2:
        return text
    if len(text) >= 7:
        return text[:3] + text[5:] + text[3:5]
    else:
        return text[:2] + text[4:] + text[2:4]
