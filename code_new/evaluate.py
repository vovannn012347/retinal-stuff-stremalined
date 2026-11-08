"""Evaluate a trained model, or predict single images.

    # metrics on a dataset folder (images/ + labels/), or only on the held-out test patients of a run
    python code_new/evaluate.py --model training-runs/surgical_gpu/final_model.pth --dataset training-data/classifier/datasets/odir_local_v1
    python code_new/evaluate.py --model training-runs/surgical_gpu/final_model.pth --dataset ... --split training-runs/surgical_gpu/split_odir_local_v1.json

    # probabilities for images
    python code_new/evaluate.py --model final_model.pth --images some_crop.jpg other.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surgical import data as D                                   # noqa: E402
from surgical.metrics import compute_metrics                     # noqa: E402
from surgical.model import load_model_file, count_params         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset')
    ap.add_argument('--split', help='split_<name>.json of a run: evaluate only its test patients')
    ap.add_argument('--images', nargs='*')
    ap.add_argument('--threshold', type=float)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    order = ckpt.get('label_order', ['glaucoma', 'atrophy', 'valid_image'])
    pp = ckpt.get('preprocess', {'size': 128, 'mode': 'stretch', 'crop_fraction': 1.0})
    thr = args.threshold if args.threshold is not None else ckpt.get('decision_threshold', 0.5)
    model = load_model_file(args.model).eval()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)
    print(f"model: {count_params(model):,} params, channels {model.channels}, hidden {model.hidden}")

    def run(paths):
        out = []
        for i in range(0, len(paths), args.batch_size):
            x = torch.stack([D.to_input(D.load_image(p, int(pp['size']), pp.get('mode', 'stretch'),
                                                     float(pp.get('crop_fraction', 1.0))))
                             for p in paths[i:i + args.batch_size]]).to(dev)
            out.append(model.predict_proba(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(order)))

    if args.images:
        for p, pr in zip(args.images, run(args.images)):
            print(Path(p).name, ' '.join(f"{n}={v:.3f}" for n, v in zip(order, pr)))

    if args.dataset:
        samples = D.scan_dataset(Path(args.dataset), order)
        if args.split:
            names = set(json.loads(Path(args.split).read_text(encoding='utf-8'))['test'])
            samples = [s for s in samples if Path(s.path).name in names]
        probs = run([s.path for s in samples])
        m = compute_metrics(probs, np.stack([s.labels for s in samples]), order, 0.5, thr)
        print(f"{m['n']} images, exact match {m['exact_match']:.3f}")
        for n in order:
            print(f"  {n:12s} acc {m[f'{n}_acc']:.3f}  recall {m[f'{n}_recall']:.3f}  "
                  f"precision {m[f'{n}_precision']:.3f}  positives {m[f'{n}_pos']}")


if __name__ == '__main__':
    main()
