def postprocess_multiline_text(text: str, count_line: int) -> str:
    if count_line == 2:
        return text[:2] + text[4:8] + text[2:4]
    return text
