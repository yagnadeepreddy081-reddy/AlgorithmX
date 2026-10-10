"""Evaluate a fixed checkpoint on synthetic or supplied real labelled data."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import datasets
from pipeline import load_model
from train import evaluate
from unet import metrics_from_confusion


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',choices=['sim','krestenitis'],default='sim')
    ap.add_argument('--split',choices=['val','test'],default='test')
    ap.add_argument('--model',default='models/unet_slick.pt')
    ap.add_argument('--out',default='out/segmentation-evaluation.json')
    args=ap.parse_args()
    model=load_model(args.model)
    ck=torch.load(args.model,map_location='cpu',weights_only=True)
    imgs,masks,_=datasets.load_split(args.split,args.source,size=ck.get('size',256))
    cm=evaluate(model,imgs,masks,'cpu');iou,precision,recall=metrics_from_confusion(cm)
    report={'source':args.source,'split':args.split,'scenes':len(imgs),'classes':datasets.CLASS_NAMES,
            'test_iou':iou,'test_precision':precision,'test_recall':recall,
            'mean_iou':sum(iou)/len(iou),'confusion':cm.tolist(),
            'checkpoint_sha256':hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            'note':'Synthetic evaluation is not field accuracy.' if args.source=='sim' else
                   'User-supplied data; verify split independence and label provenance.'}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='confusion'},indent=2))

if __name__=='__main__':main()
