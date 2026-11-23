import numpy as np


def normalized(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected a finite 2D embedding matrix")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("Zero embeddings are invalid")
    return features / norms
