import math

THRESHOLD = .90
CATALOG = {'sus scrofa': 'wild_boar', 'capra aegagrus': 'wild_goat',
           'equus quagga': 'equus_quagga', 'loxodonta africana': 'loxodonta_africana'}


def e3_decision(b_decision, e2_decision, candidate, opinion):
    # Routing provenance, not current REVIEW status, determines eligibility.
    if b_decision != 'REVIEW':
        return e2_decision, 'E2_preserved_not_original_B_REVIEW'
    assert e2_decision == 'REVIEW'
    if (opinion is not None and opinion['status'] == 'IN_CATALOG'
            and opinion['catalog_class'] == candidate and opinion['agreement']
            and math.isfinite(opinion['score']) and opinion['score'] >= THRESHOLD):
        return candidate, 'original_B_REVIEW_rescued_exact_species_score_ge_0.90'
    return 'REVIEW', 'original_B_REVIEW_not_confirmed'
