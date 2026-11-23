"""Connect the original, checksum-pinned decision functions without changing them."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from src.model_b_ood.runtime.features import ReferenceBank, assert_bank_compatible
from src.model_b_ood.runtime.gate import probabilities, decide
from src.model_b_ood.runtime.assets import ROOT as ACTIVE_ASSETS
from src.model_b_ood.runtime.rules import e2, e3, e4, e5b

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PROJECT / 'outputs/model_b_ood'
GATE = EXPERIMENTS / 'learned_gate_v1_20261103'
E2 = EXPERIMENTS / 'e2_known_veto_v1_20261103'
E3 = EXPERIMENTS / 'e3_veto_plus_original_review_v1_20261103'
E4 = EXPERIMENTS / 'e4_review_to_unknown_v1_20261103'
E5A = EXPERIMENTS / 'e5a_species_acceptance_v1_20261103'
E5B = EXPERIMENTS / 'e5b_context_125_v1_20261103'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_assets():
    """Relative paths also allow moving this adapter with its frozen local artifacts."""
    lock = read_json(Path(__file__).with_name('assets.lock.json'))
    def check(item):
        relative, expected = item
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            return relative
        return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        changed = [p for p in pool.map(check, lock['sha256'].items()) if p]
    if changed:
        raise RuntimeError('Sabit model/kural dosyası eksik veya değişmiş: ' + ', '.join(changed))
    return lock




class FrozenDecisionChain:
    def __init__(self):
        self.assets = verify_assets()
        self.model = read_json(ACTIVE_ASSETS / 'model.json')
        self.unknown_policy = read_json(ACTIVE_ASSETS / 'policy_e4.json')
        self.acceptance_policy = read_json(ACTIVE_ASSETS / 'policy_e5a.json')
        if set(self.acceptance_policy['thresholds'].values()) != {0.90}:
            raise RuntimeError('Bu adaptör kayıtlı E5-A 0.90 politikasını gerektirir.')
        with np.load(ACTIVE_ASSETS / 'reference_bank.npz', allow_pickle=False) as data:
            self.bank = ReferenceBank(data['features'], data['labels'])
        assert_bank_compatible(self.model, self.bank)
        self.e2 = e2
        self.e3 = e3
        self.e4 = e4
        self.e5b = e5b

    def score(self, vectors, crop_ids):
        if len(vectors) != len(crop_ids):
            raise ValueError('Crop kimlikleri ve embedding sayısı uyuşmuyor.')
        if not len(crop_ids):
            return []
        features, candidates, raw, _ = self.bank.transform(vectors)
        knownness, safety = probabilities(features, self.model['learned'], self.model['probability_calibration'])
        base = decide(knownness, safety, candidates, self.model['policy_B'])
        return [{'crop_id': str(crop_ids[i]), 'candidate': str(candidates[i]),
                 'knownness': float(knownness[i]), 'safety': float(safety[i]),
                 'knn_distance': float(raw['distance'][i]), 'knn_margin': float(raw['margin'][i]),
                 'B_decision': str(base[i])} for i in range(len(crop_ids))]

    def finish(self, score, pixels, image_size, opinion_fn):
        """opinion_fn(pixels,candidate,variant); never accepts truth/source labels.

        Original opinion is shared by E2/E3/E4. Context opinion is computed only
        AFTER E4 on original B REVIEW still awaiting review. It never feeds E4.
        """
        b, candidate = score['B_decision'], score['candidate']
        original = opinion_fn(pixels, candidate, 'original') if b != 'UNKNOWN' else None
        e2, e2_reason = self.e2.veto(b, original)
        e3, e3_reason = self.e3.e3_decision(b, e2, candidate, original)
        e4, e4_reason = self.e4.decide_e4(e3, original, score['knn_distance'], score['knownness'], self.unknown_policy)
        context, context_pixels = None, None
        if b == 'REVIEW' and e4 == 'REVIEW':
            context_pixels = self.e5b.geometry(pixels, image_size)
            context = opinion_fn(context_pixels, candidate, 'expanded')
        final = self.e5b.rescue(e4, b, candidate, context)
        return {**score, 'pixels': list(pixels),
                'E2_decision': e2, 'E2_reason': e2_reason,
                'E3_decision': e3, 'E3_reason': e3_reason,
                'E4_decision': e4, 'E4_reason': e4_reason,
                'E5A_decision': e4, 'E5B_decision': final,
                'context_pixels': context_pixels,
                'speciesnet_original': original, 'speciesnet_context': context,
                'speciesnet_calls': int(original is not None) + int(context is not None),
                'decision': final if final in ('REVIEW', 'UNKNOWN') else 'KNOWN',
                'species': final if final not in ('REVIEW', 'UNKNOWN') else None}
