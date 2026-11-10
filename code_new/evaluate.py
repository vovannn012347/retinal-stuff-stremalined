"""Evaluate a trained model on a dataset, or run it on images.

    # metrics on the config's first dataset (or only the held-out test patients of a run)
    python code_new/evaluate.py --config code_new/configs/definitor_gpu.yaml --model training-runs/definitor_gpu/final_model.pth
    python code_new/evaluate.py --config ... --model ... --split training-runs/definitor_gpu/split_local_definitor_v1.json

    # run on images: classifier prints probabilities, definitor writes masks to --out
    python code_new/evaluate.py --config ... --model ... --images a.jpg b.jpg --out predicted_masks
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surgical import data as D                                    # noqa: E402
from surgical.config import load_config                           # noqa: E402
from surgical.tasks import get_task, load_model_any, _TaskDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='the training config (task, preprocessing, gate)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', type=int, default=0, help='index in the config datasets list')
    ap.add_argument('--split', help='split_<name>.json of a run: evaluate only its test patients')
    ap.add_argument('--images', nargs='*')
    ap.add_argument('--out', default='predictions')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    task = get_task(cfg)
    model = load_model_any(a.model).eval()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)
    print(f"model: {type(model).__name__}, {sum(p.numel() for p in model.parameters()):,} params")

    if a.images:
        out = Path(a.out)
        for p in a.images:
            s = D.Sample(p, '', (), labels=None, mask=None)
            if task.name == 'definitor':
                img = Image.open(p).convert('RGB')
                tmp = Image.new('L', img.size)                      # dummy mask for the shared loader
                tmp_path = out / '_tmp_mask.png'
                out.mkdir(parents=True, exist_ok=True); tmp.save(tmp_path)
                s.mask = str(tmp_path)
                x, _ = task.item(task.load(s), s, False)
                prob = model.predict_proba(x[None].to(dev))[0, 0].cpu()
                Image.fromarray((prob.numpy() >= float(cfg.gate.decision_threshold)).astype('uint8') * 255).save(
                    out / (Path(p).stem + '.mask.png'))
                print(f"{Path(p).name}: disc area {float((prob >= 0.5).float().mean()):.2%} -> {out}")
                tmp_path.unlink()
            else:
                x = D.to_input(task.load(s))
                pr = model.predict_proba(x[None].to(dev))[0].cpu().tolist()
                print(Path(p).name, ' '.join(f"{n}={v:.3f}" for n, v in zip(task.order, pr)))
        return

    samples = task.scan(cfg.datasets[a.dataset])
    if a.split:
        names = set(json.loads(Path(a.split).read_text(encoding='utf-8'))['test'])
        samples = [s for s in samples if Path(s.path).name in names]
    store = D.ImageStore(task.load, cache=False)
    loader = torch.utils.data.DataLoader(_TaskDataset(samples, store, task.item, False), batch_size=a.batch_size)
    acc = task.new_accumulator()
    with torch.no_grad():
        for x, y, _ in loader:
            acc.add(model(x.to(dev)), y.to(dev))
    m = acc.result()
    print(f"{len(samples)} samples: {task.short(m)}")
    for k, v in m.items():
        if k not in ('gate_main', 'gate_recall'):
            print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    passed, _ = task.gate(m)
    print(f"gate ({'passed' if passed else 'not passed'})")


if __name__ == '__main__':
    main()
