import re
import warnings

LAST_GROUP_TO_FIRST_OFF = 0
LAST_GROUP_TO_FIRST_ON = 1


def remove_last_hyphen(string):
    if '-' in string:
        parts = string.rsplit('-', 1)
        return ''.join(parts)
    return string


def remove_first_hyphen(string):
    if '-' in string:
        return string.replace('-', '', 1)
    return string


def format_moldovan_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.replace(" ", "").upper()

    # Знаходимо всі літери та цифри
    letters = re.findall(r'[A-Z]', plate)
    digits = re.findall(r'\d', plate)

    # Якщо літери в кінці, переставляємо їх на початок
    lines = [''.join(digits), ''.join(letters)]
    return ''.join(digits + letters), lines, lines


def format_default_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.upper()
    plate = plate.replace("-", " ")
    _plate_lines = plate.split(" ")

    if len(_plate_lines) == 2:
        return plate.replace(" ", ""), _plate_lines, _plate_lines
    elif len(_plate_lines) == 1:
        warnings.warn(f"!!![WRONG COUNT LINES]!!! {plate} = {_plate_lines}")
        return plate.replace(" ", ""), _plate_lines, _plate_lines
    else:  # Якщо 3 або більше
        first_part = _plate_lines[0]
        second_part = "".join(_plate_lines[1:])
        _plate_lines = [first_part, second_part]
        return plate.replace(" ", ""), _plate_lines, _plate_lines


def format_kz_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.upper()
    _plate_lines = plate.split(" ")
    if len(_plate_lines) == 2:
        return plate.replace(" ", ""), _plate_lines, _plate_lines
    elif len(_plate_lines) == 3:
        plate = _plate_lines[0] + _plate_lines[2] + _plate_lines[1]
        _plate_lines = [_plate_lines[0], _plate_lines[2] + _plate_lines[1]]
        return plate.replace(" ", ""), _plate_lines, _plate_lines
    else:
        warnings.warn(f"!!![WRONG COUNT LINES]!!! {plate} = {_plate_lines}")
        return plate.replace(" ", ""), _plate_lines, _plate_lines


def format_ro_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.upper()
    _plate_lines = plate.split(" ")
    if len(_plate_lines) == 2:
        return plate.replace(" ", ""), _plate_lines, _plate_lines
    elif len(_plate_lines) == 3:
        _plate_lines = [_plate_lines[0] + _plate_lines[1], _plate_lines[2]]
        return plate.replace(" ", ""), _plate_lines, _plate_lines
    else:
        warnings.warn(f"!!![WRONG COUNT LINES]!!! {plate} = {_plate_lines}")
        return plate.replace(" ", ""), _plate_lines, _plate_lines


def format_al_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.replace(" ", "").upper()

    # Визначаємо шаблони для різних форматів
    patterns = [
        r'^([A-Z]{2})(\d{3})$',  # @@###
        r'^([A-Z]{2})(\d{3}[A-Z]{2})$',  # @@###@@
        r'^([A-Z]{2})(\d{2}[A-Z]{2})$',  # @@##@@
        r'^([A-Z]{2})(\d{4}[A-Z])$'  # @@####@
    ]

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            return plate, list(match.groups()), list(match.groups())

    # Якщо номер не відповідає жодному з форматів
    warnings.warn(f"!!![НЕПРАВИЛЬНИЙ ФОРМАТ]!!! {plate}")
    return plate, [plate], [plate]


def format_at_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'\s+', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    patterns = [
        (r'^([A-Z]{2})(\d{3}[A-Z]{2})$', lambda m: (m.group(1), m.group(2))),  # @@###@@
        (r'^([A-Z]{2})(\d{2}[A-Z]{2})$', lambda m: (m.group(1), m.group(2))),  # @@##@@
        (r'^([A-Z]{2})(\d{5})$', lambda m: (m.group(1), m.group(2))),  # @@#####
        (r'^([A-Z]{2})([A-Z]{2}\d{2})$', lambda m: (m.group(1), m.group(2))),  # @@@@##
        (r'^([A-Z])(\d[A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # @#@@@
        (r'^([A-Z]{2})(\d[A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # @@#@@@
        (r'^([A-Z]\d)(\d{3}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # @####@
        (r'^([A-Z]\d)(\d{4}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # @#####@
        (r'^(\d{3})([A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # ###@@@
        (r'^([A-Z]\d{2})([A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # @##@@@
        (r'^([A-Z]{2})([A-Z]{2}\d)$', lambda m: (m.group(1), m.group(2))),  # @@@@#
        (r'^(\d{4})(\d{3})$', lambda m: (m.group(1), m.group(2))),  # #######
        (r'^([A-Z]\d{3})(\d{3})$', lambda m: (m.group(1), m.group(2))),  # @######
        (r'^([A-Z]\d{2})(\d{3})$', lambda m: (m.group(1), m.group(2))),  # @#####
        (r'^(\d{2})(\d{3}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # #####@
        (r'^([A-Z])(\d{3}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # @###@

    ]

    for pattern, formatter in patterns:
        match = re.match(pattern, plate)
        if match:
            formatted = formatter(match)
            return plate, list(formatted), list(formatted)

    warnings.warn(f"!!![INVALID FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_ba_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    patterns = [
        r'^([A-Z]\d{2})(\-[A-Z]\-\d{3})$',  # @##-@-###
        r'^(\d{3}\-[A-Z])(\-\d{3})$',       # ###-@-###
    ]

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_be_plate(plate, count_line, *args, **kwargs):
    plate = remove_last_hyphen(plate)

    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    patterns = [
        r'^([A-Z]{3})(\d{3})$',  # @@@###
        r'^([A-Z]\-[A-Z]{3})(\d{3})$',       # @-@@@###
        r'^(\d\-[A-Z]{3})(\d{3})$',  # #-@@@###
    ]

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_bg_plate(plate, count_line, *args, **kwargs):
    plate = remove_last_hyphen(plate)

    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    patterns = [
        r'^([A-Z]{2})(\d{4})([A-Z]{2})$',  # @@###@@
        r'^([A-Z]{2})(\d{4})([A-Z])$',     # @@###@
        r'^([A-Z])(\d{4})([A-Z]{2})$',  # @###@@

    ]

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())
            punctuated_lines = [punctuated_lines[0]+punctuated_lines[2], punctuated_lines[1]]

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_cy_plate(plate, count_line, *args, **kwargs):
    plate = remove_last_hyphen(plate)

    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    patterns = [
        r'^([A-Z]{3})(\d{3})$',  # @@@###

    ]

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_dk_plate(plate, count_line, *args, **kwargs):
    plate = remove_last_hyphen(plate)

    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+\.]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    patterns = [
        r'^([A-Z]{2}\d{2})(\d{3})$',  # @@## ###
        r'^([A-Z]\d{2})(\d{3})$',  # @## ###
        r'^([A-Z]{2})(\d{3})$',  # @@ ###

    ]

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_es_plate(plate, count_line, *args, **kwargs):
    #plate = remove_last_hyphen(plate)

    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+\.]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    if count_line == 3:
        patterns = [
            (r'^([A-Z]\d)(\d{3})([A-Z]{3})$', LAST_GROUP_TO_FIRST_OFF, ""),  # @# ### @@@

        ]
    else:
        patterns = [
            (r'^([A-Z]{2})(\d{4})([A-Z])$', LAST_GROUP_TO_FIRST_ON, "-"),     #: @@ #### @
            (r'^([A-Z])(\d{4})([A-Z]{2})$', LAST_GROUP_TO_FIRST_ON, "-"),     #: @ #### @@
            (r'^([A-Z])(\d{4})([A-Z])$', LAST_GROUP_TO_FIRST_ON, "-"),        #: @ #### @
            (r'^([A-Z]\d{4})([A-Z]{3})$', LAST_GROUP_TO_FIRST_OFF, ""),  #: @ #### @@@
            (r'^([A-Z]\d{3})(\d{3})$', LAST_GROUP_TO_FIRST_OFF, ""),  #: @### ###
            (r'^([A-Z]\-\d{3})(\d{3})$', LAST_GROUP_TO_FIRST_OFF, ""),  #: @-### ###

        ]

    for pattern, last_group_to_first, replacer in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())
            if last_group_to_first:
                punctuated_lines = [punctuated_lines[0] + replacer + punctuated_lines[2], punctuated_lines[1]]

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_gg_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+\.]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    if count_line == 1:
        patterns = [
            (r'^\d{2}\d{2}$', LAST_GROUP_TO_FIRST_OFF, "", 0),  #: ####
        ]
    elif count_line == 2:
        patterns = [
            (r'^(\d{2})(\d{3})$', LAST_GROUP_TO_FIRST_OFF, "", 1),  #: ## ###
            (r'^(\d{2})(\d{2})$', LAST_GROUP_TO_FIRST_OFF, "", 0),  #: ## ##
        ]
    else:
        patterns = []

    for pattern, last_group_to_first, replacer, multistate in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())
            if last_group_to_first:
                punctuated_lines = [punctuated_lines[0] + replacer + punctuated_lines[2], punctuated_lines[1]]

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            if multistate:
                parsed_lines[0] = [parsed_lines[0], parsed_lines[0]+parsed_lines[1][0]]
                parsed_lines[1] = [parsed_lines[1], parsed_lines[1][1:]]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_gr_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+\.\-]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    if count_line == 1:
        patterns = []
    elif count_line == 2:
        patterns = [
            r'^([A-Z]{3})(\d{3})$',  #: @@@ ###
            r'^([A-Z]{3})(\d{2})$',  #: @@@ ##
            r'^([A-Z]{3})(\d{4})$',  #: @@@ ###
        ]
    else:
        patterns = []

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())

            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_is_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+\.]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    if count_line == 3:
        patterns = []
    elif count_line == 2:
        patterns = [
            # невтйде усюди проставити дефіс для цього формату - є для авто, немає для мото
            (r'^([A-Z]{2})(\d{3})$', LAST_GROUP_TO_FIRST_OFF, "", 1),     #: @@ ###
            (r'^([A-Z]{2})([A-Z]\d{2})$', LAST_GROUP_TO_FIRST_ON, "-", 1),  #: @@ @##

        ]
    else:
        patterns = []

    for pattern, last_group_to_first, replacer, is_change_places in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())
            if last_group_to_first:
                punctuated_lines = [punctuated_lines[0] + replacer, punctuated_lines[1]]
            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            if is_change_places:
                parsed_lines = [[parsed_lines[1], parsed_lines[0]],
                                [parsed_lines[1], parsed_lines[0]]]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_li_plate(plate, count_line, *args, **kwargs):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+\.]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    if count_line == 3:
        patterns = []
    elif count_line == 2:
        patterns = [
            # невтйде усюди проставити дефіс для цього формату - є для авто, немає для мото
            (r'^([A-Z]{2})(\d{5})$', LAST_GROUP_TO_FIRST_OFF, "", 0),     #: @@ #####
            (r'^([A-Z]{2})(\d{4})$', LAST_GROUP_TO_FIRST_OFF, "", 0),  #: @@ ####

        ]
    else:
        patterns = []

    for pattern, last_group_to_first, replacer, is_change_places in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())
            if last_group_to_first:
                punctuated_lines = [punctuated_lines[0] + replacer, punctuated_lines[1]]
            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            if is_change_places:
                parsed_lines = [[parsed_lines[1], parsed_lines[0]],
                                [parsed_lines[1], parsed_lines[0]]]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


def format_nl_plate(plate, count_line, *args, **kwargs):
    plate = remove_first_hyphen(plate)

    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'[\s+\.]', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    if count_line == 3:
        patterns = []
    elif count_line == 2:
        patterns = [
            # невтйде усюди проставити дефіс для цього формату - є для авто, немає для мото
            (r'^(\d{2})([A-Z]{3})\-(\d)$', LAST_GROUP_TO_FIRST_ON, "-", 0),        #: ## @@@-#
            (r'^(\d)([A-Z]{3})\-(\d{2})$', LAST_GROUP_TO_FIRST_ON, "-", 1),        #: #-@ @@-##
            (r'^(\d)([A-Z])(([A-Z]){2})\-(\d{2})$', LAST_GROUP_TO_FIRST_ON, "-", 0),     #: @ ###-@@
            (r'^([A-Z]{2})(\d{3})\-([A-Z])$', LAST_GROUP_TO_FIRST_ON, "-", 0),     #: @@ ###-@
            (r'^(\d{2})(\d{2})\-([A-Z]{2})$', LAST_GROUP_TO_FIRST_ON, "-", 0),     #: ## ##-@@
            (r'^(\d{2})([A-Z]{2})\-([A-Z]{2})$', LAST_GROUP_TO_FIRST_ON, "-", 0),  #: ## @@-@@
            (r'^(\d{2})([A-Z]{2})\-(\d{2})$', LAST_GROUP_TO_FIRST_ON, "-", 0),     #: ## @@-##
            (r'^([A-Z]{2})([A-Z]{2})\-(\d{2})$', LAST_GROUP_TO_FIRST_ON, "-", 0),  #: @@ @@-##
            (r'^([A-Z]{2})(\d{2})\-(\d{2})$', LAST_GROUP_TO_FIRST_ON, "-", 0),     #: @@ ##-##
            (r'^([A-Z]{2})(\d{2})\-([A-Z]{2})$', LAST_GROUP_TO_FIRST_ON, "-", 0),  #: @@ ##-@@

        ]
    else:
        patterns = []

    for pattern, last_group_to_first, replacer, is_change_places in patterns:
        match = re.match(pattern, plate)
        if match:
            clean_plate = plate
            punctuated_lines = list(match.groups())
            if last_group_to_first and len(punctuated_lines) == 4:
                punctuated_lines = [punctuated_lines[0] + replacer + punctuated_lines[1],
                                    punctuated_lines[2] + replacer + punctuated_lines[3]]
            elif last_group_to_first:
                punctuated_lines = [punctuated_lines[0], punctuated_lines[1] + replacer + punctuated_lines[2]]
            parsed_lines = [l.replace("-", "") for l in punctuated_lines]
            return clean_plate, parsed_lines, punctuated_lines

    warnings.warn(f"!!![WRONG FORMAT]!!! {plate}")
    return plate, [plate], [plate]


fromats_parse = {
    "md": format_moldovan_plate,
    "kz": format_kz_plate,
    'ro': format_ro_plate,
    "default": format_default_plate,
    "fi": format_default_plate,
    "al": format_al_plate,
    "at": format_at_plate,
    "ba": format_ba_plate,
    "be": format_be_plate,
    "bg": format_bg_plate,
    "cy": format_cy_plate,
    "de": format_default_plate,
    "dk": format_dk_plate,
    "es": format_es_plate,
    "gg": format_gg_plate,
    "gr": format_gr_plate,
    "is": format_is_plate,
    "li": format_li_plate,
    "lu": format_default_plate,
    "mt": format_default_plate,
    "nl": format_nl_plate,
    "no": format_default_plate,
    "pl": format_default_plate,
    "uk": format_default_plate,
}
