from pathlib import Path
import hashlib
import json

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / 'models/ood/e5b_v1'
BIOCLIP = ROOT / 'bioclip'

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
