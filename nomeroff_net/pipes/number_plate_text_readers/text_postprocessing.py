
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
