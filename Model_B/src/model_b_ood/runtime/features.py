from __future__ import annotations

import hashlib
import numpy as np

from .neighbors import normalized


class ReferenceBank:
    """Frozen encoder bank plus per-class prototype/diagonal distribution summaries."""
    def __init__(self, vectors, labels, k=5):
        self.vectors = normalized(vectors)
        self.labels = np.asarray(labels, dtype=str)
        self.classes = np.unique(self.labels)
        self.k = k
        self.prototypes, self.variance = [], []
        for name in self.classes:
            values = self.vectors[self.labels == name]
            if len(values) < max(k, 2):
                raise ValueError('Too few class references')
            self.prototypes.append(values.mean(axis=0))
            variance = values.var(axis=0)
            self.variance.append(.9 * variance + .1 * variance.mean() + 1e-6)
        self.prototypes = np.asarray(self.prototypes)
        self.variance = np.asarray(self.variance)
        self.fingerprint = hashlib.sha256(self.vectors.tobytes() + '\n'.join(self.labels).encode()).hexdigest()

    def transform(self, vectors):
        query = normalized(vectors)
        features, predictions, raw = [], [], []
        for start in range(0, len(query), 128):
            x = query[start:start + 128]
            similarity = x @ self.vectors.T
            class_scores = np.stack([np.partition(similarity[:, self.labels == c], -self.k, axis=1)[:, -self.k:].mean(axis=1) for c in self.classes], axis=1)
            order = np.argsort(class_scores, axis=1)
            best, second = order[:, -1], order[:, -2]
            best_score = class_scores[np.arange(len(x)), best]
            second_score = class_scores[np.arange(len(x)), second]
            nn_ids = np.argsort(similarity, axis=1)[:, -10:]
            nn = np.take_along_axis(similarity, nn_ids, axis=1)
            vote = np.stack([(self.labels[nn_ids] == c).mean(axis=1) for c in self.classes], axis=1)
            entropy = -(vote * np.log(np.maximum(vote, 1e-8))).sum(axis=1)
            prototypes = normalized(self.prototypes)
            proto_cos = x @ prototypes.T
            mahal = np.stack([np.sqrt(np.mean((x - p) ** 2 / v, axis=1)) for p, v in zip(self.prototypes, self.variance)], axis=1)
            summary = np.stack([best_score, second_score, best_score - second_score,
                                nn[:, -1], nn[:, -5:].mean(axis=1), nn[:, -5], nn[:, -5:].std(axis=1),
                                vote[np.arange(len(x)), best], entropy,
                                proto_cos[np.arange(len(x)), best], np.sort(proto_cos, axis=1)[:, -1],
                                np.sort(proto_cos, axis=1)[:, -1] - np.sort(proto_cos, axis=1)[:, -2],
                                mahal[np.arange(len(x)), best], mahal.min(axis=1)], axis=1)
            # Class-specific evidence explicitly changes schema if catalog changes.
            features.append(np.concatenate([summary, 1 - class_scores, proto_cos, mahal], axis=1))
            predictions.extend(self.classes[best])
            raw.append(np.stack([np.clip(1 - best_score, 0, 2), best_score - second_score], axis=1))
        matrix = np.concatenate(features)
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite gate features')
        names = ['class_top1', 'class_top2', 'margin', 'nn1', 'nn5_mean', 'nn5_kth', 'nn5_std', 'vote_for_candidate',
                 'vote_entropy', 'candidate_prototype_cos', 'max_prototype_cos', 'prototype_margin', 'candidate_diag_distance', 'min_diag_distance']
        names += [f'{metric}:{c}' for metric in ('class_distance', 'prototype_cos', 'diag_distance') for c in self.classes]
        raw = np.concatenate(raw)
        return matrix, np.array(predictions), {'distance': raw[:, 0], 'margin': raw[:, 1], 'predicted': np.array(predictions)}, names


def assert_bank_compatible(model_info, bank):
    if model_info['bank_fingerprint'] != bank.fingerprint or model_info['classes'] != bank.classes.tolist():
        raise ValueError('Reference bank/catalog changed: refit gate and recalibrate before use')
