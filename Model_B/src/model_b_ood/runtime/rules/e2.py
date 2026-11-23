import math

CATALOG = {'sus scrofa': 'wild_boar', 'capra aegagrus': 'wild_goat',
           'equus quagga': 'equus_quagga', 'loxodonta africana': 'loxodonta_africana'}
THRESHOLD = 0.90


def is_known_decision(decision):
    return decision not in ('REVIEW', 'UNKNOWN')


def veto(decision, opinion):
    if not is_known_decision(decision):
        return decision, 'B_' + decision + '_unchanged'
    if opinion is None:
        return 'REVIEW', 'missing_second_opinion'
    if opinion['status'] != 'IN_CATALOG':
        return 'REVIEW', opinion['status']
    if opinion['catalog_class'] != decision:
        return 'REVIEW', 'different_catalog_species'
    if not math.isfinite(opinion['score']) or opinion['score'] < THRESHOLD:
        return 'REVIEW', 'agreement_below_0.90'
    return decision, 'exact_species_agreement_score_ge_0.90'
