"""Image -> YOLO boxes -> frozen crop OOD chain -> PNG and JSON."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .chain import PROJECT, EXPERIMENTS, FrozenDecisionChain, sha256

OUTPUT_ROOT = PROJECT / 'outputs/phototrap_ood'
YOLO_WEIGHTS = PROJECT / 'models/multispecies_v0.3_demo/best.pt'
# Existing detector demonstration settings; no detector threshold search.
DETECTOR_CONFIG = {'conf': 0.25, 'iou': 0.7, 'imgsz': 640, 'max_det': 300, 'quantize': None}
LABELS = {'wild_boar': 'Yaban domuzu', 'wild_goat': 'Yaban keçisi',
          'equus_quagga': 'Zebra', 'loxodonta_africana': 'Afrika fili'}


def new_output(path=None, prefix='predict'):
    if path is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        category = 'validation' if prefix.startswith('verification') else 'predictions'
        path = OUTPUT_ROOT / category / (prefix + '_' + stamp + '_' + uuid.uuid4().hex[:6])
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Yeni çıktılar outputs/phototrap_ood altında olmalı.')
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def pixel_box(xyxy, size):
    width, height = size
    if len(xyxy) != 4 or not all(math.isfinite(float(v)) for v in xyxy):
        raise ValueError('Geçersiz YOLO kutusu.')
    x1,y1,x2,y2 = xyxy
    result = [max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(width, math.ceil(x2)), min(height, math.ceil(y2))]
    if not (result[0] < result[2] and result[1] < result[3]):
        raise ValueError('YOLO kutusu görüntü içinde geçerli piksel içermiyor.')
    return result


class PhotoRuntime:
    def __init__(self, device='auto'):
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA kullanılamıyor; --device cpu ile çalıştırın.')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.chain = FrozenDecisionChain()
        self.detector = None
        self.encoder = None
        self.preprocess = None
        self.speciesnet = None

    def detect(self, image, output):
        from ultralytics import YOLO
        if self.detector is None:
            self.detector = YOLO(str(YOLO_WEIGHTS))
        # Pass stored pixels as BGR ndarray: no path reader or implicit EXIF rotation.
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        result = self.detector.predict(source=bgr, device=self.device, verbose=False,
                   save=False, save_txt=False, save_crop=False, augment=False,
                   project=str(output), name='detector', **DETECTOR_CONFIG)[0]
        boxes = []
        if result.boxes is not None:
            for i, (xyxy, confidence, cls) in enumerate(zip(result.boxes.xyxy.cpu().tolist(),
                       result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())):
                boxes.append({'detection_id': f'box_{i+1:03d}', 'xyxy': xyxy,
                              'pixels': pixel_box(xyxy,image.size),
                              'detector_class': result.names[int(cls)],
                              'detector_confidence': float(confidence)})
        # The small detector need not occupy GPU memory while the two encoders run.
        self.detector.to('cpu')
        return boxes

    def embed(self, image, pixels, batch_size=16):
        import torch
        from src.model_b_ood.runtime.bioclip import load_encoder
        from src.model_b_ood.runtime.assets import BIOCLIP
        if not pixels:
            return np.empty((0,512), dtype=np.float32)
        if self.encoder is None:
            self.encoder, self.preprocess, _ = load_encoder(BIOCLIP,self.device)
        values=[]
        with torch.inference_mode():
            for start in range(0,len(pixels),batch_size):
                tensor=torch.stack([self.preprocess(image.crop(box)) for box in pixels[start:start+batch_size]]).to(self.device)
                values.append(self.encoder.encode_image(tensor,normalize=True).float().cpu().numpy())
        return np.concatenate(values)

    def opinion(self, image, pixels, candidate, variant):
        from src.model_b_ood.runtime.speciesnet import SpeciesNetOpinion
        if self.speciesnet is None:
            self.speciesnet=SpeciesNetOpinion(device=self.device)
        return self.speciesnet.predict(image,pixels,candidate)

    def predict(self, image_path, output=None):
        image_path=Path(image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError('Giriş fotoğrafı bulunamadı: '+str(image_path))
        # Same stored-pixel policy as the experiments; keep EXIF orientation explicit.
        with Image.open(image_path) as source:
            image=source.convert('RGB')
        output=new_output(output)
        started=time.perf_counter()
        detections=self.detect(image,output)
        detector_seconds=time.perf_counter()-started
        records=[]
        ood_started=time.perf_counter()
        if detections:
            vectors=self.embed(image,[r['pixels'] for r in detections])
            scores=self.chain.score(vectors,[r['detection_id'] for r in detections])
            for detection,score in zip(detections,scores):
                record=self.chain.finish(score,detection['pixels'],image.size,
                         lambda pixels,candidate,variant:self.opinion(image,pixels,candidate,variant))
                records.append({**detection,**record})
        counts=Counter(r['decision'] for r in records)
        result={'schema_version':1,'input_path':str(image_path),'input_sha256':sha256(image_path),
                'input_size':list(image.size),'status':'detections' if records else 'no_detections',
                'message':'Tespit tamamlandı.' if records else 'Tespit yok.',
                'summary':{k:counts[k] for k in ['KNOWN','UNKNOWN','REVIEW']},
                'detection_count':len(records),'box_source':'YOLO_v0.3','boxes':records,
                'detector':{'weights':str(YOLO_WEIGHTS),'configuration':DETECTOR_CONFIG,'device':self.device},
                'ood':{'chain':'B -> E2 -> E3 -> E4 -> E5-B; E5-A keeps 0.90',
                       'device':self.device,'bank_fingerprint':self.chain.bank.fingerprint,
                       'crop_policy':'floor/ceil + image clip; stored pixels, no EXIF rotation; centered 1.25 context only on original B REVIEW still REVIEW after E4',
                       'speciesnet_calls':sum(r['speciesnet_calls'] for r in records),
                       'bioclip_crops':len(records),'asset_sha256':self.chain.assets['sha256']},
                'timing':{'detector_seconds':detector_seconds,'ood_seconds':time.perf_counter()-ood_started,
                          'total_seconds':time.perf_counter()-started},
                'outputs':{'image':str(output/'result.png'),'json':str(output/'result.json')}}
        render(image,records,result['summary'],output/'result.png')
        save_json(output/'result.json',result)
        image.close()
        return result


def render(image, records, summary, path):
    font_size=max(15,min(36,round(image.width/65)))
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',font_size)
    except OSError:
        font=ImageFont.load_default(size=font_size)
    footer=font_size*3+20
    canvas=Image.new('RGB',(image.width,image.height+footer),'#111827')
    canvas.paste(image,(0,0))
    draw=ImageDraw.Draw(canvas)
    colors={'KNOWN':'#22c55e','UNKNOWN':'#ef4444','REVIEW':'#f59e0b'}
    for r in records:
        pixels=r['pixels'];color=colors[r['decision']]
        draw.rectangle((pixels[0],pixels[1],min(pixels[2],image.width-1),min(pixels[3],image.height-1)),outline=color,width=max(2,font_size//7))
        label=r['decision']+(' | '+LABELS.get(r['species'],r['species']) if r['species'] else '')
        label+=f" | YOLO {r['detector_confidence']:.2f}"
        bounds=draw.textbbox((0,0),label,font=font)
        left=max(0,min(pixels[0],image.width-(bounds[2]+12)))
        top=max(0,pixels[1]-font_size-12)
        draw.rectangle((left,top,min(image.width,left+bounds[2]+12),top+font_size+12),fill='#111827')
        draw.text((left+5,top+2),label,font=font,fill=color)
    text='Tespit yok.' if not records else ' | '.join(f'{key}: {summary[key]}' for key in ['KNOWN','UNKNOWN','REVIEW'])
    draw.text((12,image.height+8),text,font=font,fill='white')
    draw.text((12,image.height+font_size+20),'Fotokapan + OOD prototipi',font=font,fill='#cbd5e1')
    if Path(path).exists():
        raise FileExistsError(path)
    canvas.save(path)
