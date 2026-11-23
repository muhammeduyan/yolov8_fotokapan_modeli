import math

CATALOG = {'sus scrofa': 'wild_boar', 'capra aegagrus': 'wild_goat',
           'equus quagga': 'equus_quagga', 'loxodonta africana': 'loxodonta_africana'}


def scientific(label):
    parts = label.split(';')
    if len(parts) < 7 or not parts[4].strip() or not parts[5].strip():
        return None
    return parts[4].strip().lower() + ' ' + parts[5].strip().lower()


def outside_species(opinion):
    if opinion is None:
        return False
    parts = opinion['raw_label'].split(';')
    taxon = scientific(opinion['raw_label'])
    return (parts[-1].strip().lower() != 'blank' and taxon is not None and taxon not in CATALOG
            and opinion['status'] == 'OUTSIDE_CATALOG')


def decide_e4(current, opinion, distance, knownness, policy):
    # This function has no truth label, source, species or crop-ID arguments.
    if current != 'REVIEW':
        return current, 'E3_non_REVIEW_preserved'
    if not policy['enabled']:
        return 'REVIEW', 'E4_disabled_by_calibration'
    if not outside_species(opinion):
        return 'REVIEW', 'no_full_outside_species_evidence'
    if not all(math.isfinite(x) for x in [opinion['score'], distance, knownness]):
        return 'REVIEW', 'nonfinite_evidence'
    if opinion['score'] < policy['sn_min_score']:
        return 'REVIEW', 'outside_species_confidence_too_low'
    if distance < policy['knn_min_distance']:
        return 'REVIEW', 'KNN_too_close_to_known_bank'
    if knownness > policy['knownness_max']:
        return 'REVIEW', 'gate_knownness_too_high'
    return 'UNKNOWN', 'outside_species_AND_score_AND_KNN_distance_AND_knownness'
