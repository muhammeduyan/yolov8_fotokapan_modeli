import math

CATALOG = {'sus scrofa': 'wild_boar', 'capra aegagrus': 'wild_goat',
           'equus quagga': 'equus_quagga', 'loxodonta africana': 'loxodonta_africana'}


def scientific(label):
    p = label.split(';')
    return p[4].strip().lower() + ' ' + p[5].strip().lower() if len(p) >= 7 and p[4].strip() and p[5].strip() else None


def geometry(pixels, size):
    x1, y1, x2, y2 = pixels
    width, height = size
    # Nominal linear scale 1.25; outward pixel rounding is symmetric.
    reqx, reqy = math.ceil((x2 - x1) / 8), math.ceil((y2 - y1) / 8)
    padx, pady = min(reqx, x1, width - x2), min(reqy, y1, height - y2)
    return [x1 - padx, y1 - pady, x2 + padx, y2 + pady]


def rescue(current, b_decision, candidate, opinion):
    if current != 'REVIEW' or b_decision != 'REVIEW':
        return current
    if (opinion is not None and opinion['status'] == 'IN_CATALOG'
            and CATALOG.get(scientific(opinion['raw_label'])) == candidate
            and opinion['catalog_class'] == candidate and math.isfinite(opinion['score']) and opinion['score'] >= .90):
        return candidate
    return 'REVIEW'
