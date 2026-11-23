from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser(description='Sabit YOLO v0.3 + E2–E5-B fotokapan prototipi')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('predict',help='Bir fotoğrafı tespit et; kutulu PNG ve JSON kaydet')
    p.add_argument('image',type=Path)
    p.add_argument('--output',type=Path,help='Yeni ve mevcut olmayan outputs/phototrap_ood alt klasörü')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p=sub.add_parser('verify',help='Kayıtlı 857 crop için entegrasyon eşleşmesini doğrula; eşik seçmez')
    p.add_argument('--fresh',action='store_true',help='Önbellekli eşleşmeden sonra tüm 857 crop için yeni BioCLIP ve SpeciesNet çıkarımı da yap')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if args.command=='predict':
        from .runtime import PhotoRuntime
        result=PhotoRuntime(args.device).predict(args.image,args.output)
        print(result['message'])
        print('Kararlar:',result['summary'])
        print('Görsel:',result['outputs']['image'])
        print('JSON:',result['outputs']['json'])
    else:
        from .validation.replay import verify
        verify(args.output,args.fresh,args.device)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('Çalıştırma hatası: '+str(exc),file=sys.stderr)
        sys.exit(1)
