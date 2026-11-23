from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .assets import sha256
from .assets import ROOT, read_json

CATALOG = {'sus scrofa': 'wild_boar', 'capra aegagrus': 'wild_goat', 'equus quagga': 'equus_quagga', 'loxodonta africana': 'loxodonta_africana'}


def interpret(label):
    parts = label.split(';')
    if len(parts) < 7:
        return {'status': 'TAXON_GROUP', 'catalog_class': None, 'taxon': label}
    genus, species = parts[4].strip().lower(), parts[5].strip().lower()
    if not genus or not species:
        return {'status': 'TAXON_GROUP', 'catalog_class': None, 'taxon': label}
    scientific = genus + ' ' + species
    if scientific in CATALOG:
        return {'status': 'IN_CATALOG', 'catalog_class': CATALOG[scientific], 'taxon': scientific}
    return {'status': 'OUTSIDE_CATALOG', 'catalog_class': None, 'taxon': scientific}


class SpeciesNetOpinion:
    """Official 4.0.3a crop weights, classifier-only inference, no detector download.

    Reproduces Google's classifier preprocessing: resize float crop without
    antialiasing, quantize uint8, HWC /255. Model itself expects NHWC.
    """
    def __init__(self, device='cuda'):
        import torch
        sys.path.insert(0, str(ROOT / 'vendor'))
        import onnx2torch  # required by the official checkpoint's serialized graph
        torch.set_num_threads(2)
        info = read_json(ROOT / 'speciesnet_model/info.json')
        path = ROOT / 'speciesnet_model' / info['classifier']
        expected = read_json(ROOT / 'speciesnet_files.json')[path.name]['sha256']
        if sha256(path) != expected:
            raise ValueError('SpeciesNet checkpoint checksum mismatch')
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(path, map_location=self.device, weights_only=False).eval()
        self.labels = (ROOT / 'speciesnet_model' / info['classifier_labels']).read_text(encoding='utf-8').splitlines()
        self.calls = 0

    def predict(self, image, pixels, candidate):
        import torch
        import torchvision.transforms.functional as F
        started = time.perf_counter()
        crop = image.convert('RGB').crop(tuple(pixels))
        tensor = F.convert_image_dtype(F.pil_to_tensor(crop), torch.float32)
        tensor = F.resize(tensor, [480, 480], antialias=False)
        arr = F.convert_image_dtype(tensor, torch.uint8).permute(1, 2, 0).numpy()
        batch = torch.from_numpy(np.asarray(arr / 255., dtype=np.float32)[None]).to(self.device)
        with torch.inference_mode():
            scores = torch.softmax(self.model(batch), dim=-1)[0]
            values, indices = torch.topk(scores, 5)
        labels = [self.labels[int(i)] for i in indices.cpu()]
        result = interpret(labels[0])
        self.calls += 1
        return {**result, 'raw_label': labels[0], 'score': float(values[0].cpu()),
                'top5_labels': labels, 'top5_scores': values.cpu().tolist(),
                'agreement': result['catalog_class'] == candidate,
                'seconds': time.perf_counter() - started, 'device': self.device,
                'version': '4.0.3a', 'verified_new_species': False}


def fuse(decision, candidate, opinion, threshold):
    # Only an already-routed REVIEW can be rescued. Outside proposals never become labels.
    if decision != 'REVIEW' or opinion is None:
        return decision
    if opinion['status'] == 'IN_CATALOG' and opinion['agreement'] and opinion['score'] >= threshold:
        return candidate
    return 'REVIEW'
