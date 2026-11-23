import json
from pathlib import Path
from .assets import sha256


def load_encoder(model_dir: Path, device: str):
    """Load a downloaded, revision-pinned OpenCLIP model without network access."""
    import open_clip
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    source = json.loads((model_dir / 'source.json').read_text(encoding='utf-8'))
    if sha256(model_dir / 'open_clip_pytorch_model.bin') != source['weights_sha256']:
        raise ValueError('BioCLIP weights checksum mismatch')
    model, _, preprocess = open_clip.create_model_and_transforms(
        "local-dir:" + str(model_dir.resolve()), device=device)
    model.eval()
    return model, preprocess, device
