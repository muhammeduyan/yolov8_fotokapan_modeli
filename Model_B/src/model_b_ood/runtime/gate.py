import numpy as np
from scipy.special import expit


def logit(features, model):
    return ((np.asarray(features) - model['mean']) / model['scale']) @ np.asarray(model['coef']) + model['intercept']


def probabilities(features, model, calibration):
    p = {key: expit(logit(logit(features, model[key])[:, None], calibration[key])) for key in model}
    return p['knownness'], np.minimum(p['knownness'], p['safety'])


def decide(knownness, safety, predicted, policy):
    result = np.full(len(safety), 'REVIEW', dtype=object)
    accepted = (safety >= policy['accept_safety_threshold']) & policy['auto_accept_enabled']
    result[(knownness < policy['reject_knownness_threshold']) & ~accepted] = 'UNKNOWN'
    result[accepted] = np.asarray(predicted)[accepted]
    return result.astype(str)
