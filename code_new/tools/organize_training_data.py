"""Arrange training-data/ for the SurgicalSlicing trainer and move old model files away.

Shows the plan by default; nothing changes until --apply.

    python code_new/tools/organize_training_data.py               # dry run: print the plan
    python code_new/tools/organize_training_data.py --apply       # do it (writes an undo log)
    python code_new/tools/organize_training_data.py --undo training-data/organize_log_<time>.json

What it does
  1. Old model files -> old-training-data/<same relative path>   (MOVE)
       *.pth / *.pt / *.ckpt, TensorBoard event files, everything under a `checkpoints`
       folder (training snapshots, logs), and retina-stuff-definitor/for-display (old run logs).
  2. Classifier datasets -> training-data/classifier/datasets/<name>/{images,labels}   (COPY)
       odir_local_v1  <- artifacts/retina_images_prepared/retina_classify/nerves_cropped_out(+_labels)
                         (the set the 16_01_2026 trainer used: optic-disc crops + per-image label csv)
       legacy_v0      <- training-data/retina-stuff-classifier (incorrect crops + labels) and the
                         definitor crops training-data/retina-stuff-definitor/nerves_defined_output
                         matched to those labels by name (X_cropped_0.jpg <-> X.jpg.csv)
     Sources are copied, not moved, so the old prepare scripts keep working; delete them later if
     you like. Everything else in training-data/ (detection, infill, raw tables) stays where it is.
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TD = REPO / 'training-data'
OLD = REPO / 'old-training-data'
DS = TD / 'classifier' / 'datasets'
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.jfif')
MODEL_EXT = ('.pth', '.pt', '.ckpt')


def is_img(p: Path):
    return p.suffix.lower() in IMG_EXT


def plan_old_models():
    moves = []
    if not TD.is_dir():
        return moves
    for p in sorted(TD.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(TD)
        parts = [s.lower() for s in rel.parts]
        if rel.parts[0] == 'classifier':
            continue
        old = (p.suffix.lower() in MODEL_EXT or p.name.startswith('events.out.tfevents')
               or 'checkpoints' in parts[:-1]
               or (len(parts) > 2 and parts[0] == 'retina-stuff-definitor' and parts[1] == 'for-display'))
        if old:
            moves.append((p, OLD / rel))
    return moves


def plan_odir_local():
    src_img = REPO / 'artifacts' / 'retina_images_prepared' / 'retina_classify' / 'nerves_cropped_out'
    src_lbl = REPO / 'artifacts' / 'retina_images_prepared' / 'retina_classify' / 'nerves_cropped_out_labels'
    dst = DS / 'odir_local_v1'
    copies, notes = [], []
    if not src_img.is_dir() or not src_lbl.is_dir():
        notes.append(f"odir_local_v1: source not found ({src_img}) - skipped")
        return copies, notes
    n_missing = 0
    for p in sorted(src_img.rglob('*')):
        if not p.is_file() or not is_img(p):
            continue
        lbl = src_lbl / (p.name + '.csv')
        if not lbl.is_file():
            n_missing += 1
            continue
        copies.append((p, dst / 'images' / p.name))
        copies.append((lbl, dst / 'labels' / lbl.name))
    notes.append(f"odir_local_v1: {len(copies) // 2} labelled images"
                 + (f", {n_missing} without labels skipped" if n_missing else ''))
    return copies, notes


def plan_legacy():
    cls = TD / 'retina-stuff-classifier'
    lbl_dir = cls / 'nerves_defined_output_labels'
    bad_dir = cls / 'nerves_defined_output_1_iccorect'
    crops_dir = TD / 'retina-stuff-definitor' / 'nerves_defined_output'
    dst = DS / 'legacy_v0'
    copies, notes = [], []
    if not lbl_dir.is_dir():
        notes.append("legacy_v0: no label folder - skipped")
        return copies, notes
    labels = {p.name[:-4]: p for p in lbl_dir.glob('*.csv')}            # 'X.jpg' -> csv
    labels_by_stem = {Path(k).stem: v for k, v in labels.items()}
    used, n_bad, n_crop, n_crop_skip = set(), 0, 0, 0
    bad_stems = set()
    if bad_dir.is_dir():
        for p in sorted(bad_dir.iterdir()):
            if p.is_file() and is_img(p) and p.name in labels:
                copies.append((p, dst / 'images' / p.name))
                copies.append((labels[p.name], dst / 'labels' / (p.name + '.csv')))
                used.add(p.name); bad_stems.add(p.stem); n_bad += 1
    if crops_dir.is_dir():
        rx = re.compile(r'^(.*)_cropped_(\d+)$')
        for p in sorted(crops_dir.iterdir()):
            if not (p.is_file() and is_img(p)):
                continue
            m = rx.match(p.stem)
            if not m:
                continue
            stem = m.group(1)
            if stem in bad_stems:            # this crop is the one marked incorrect
                n_crop_skip += 1
                continue
            lbl = labels_by_stem.get(stem)
            if lbl is None:
                continue
            copies.append((p, dst / 'images' / p.name))
            copies.append((lbl, dst / 'labels' / (p.name + '.csv')))
            used.add(lbl.name[:-4]); n_crop += 1
    unused = len([k for k in labels if k not in used])
    notes.append(f"legacy_v0: {n_bad} incorrect crops + {n_crop} definitor crops matched to labels "
                 f"({n_crop_skip} crops skipped because the same eye is in the incorrect set, "
                 f"{unused} label files without an image)")
    return copies, notes


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def apply(moves, copies):
    log = {'created': datetime.now().isoformat(timespec='seconds'), 'moves': [], 'copies': []}
    log_path = TD / f"organize_log_{datetime.now():%Y%m%d_%H%M%S}.json"
    try:
        for src, dst in moves:
            if dst.exists():
                print(f"  skip move (target exists): {dst}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            log['moves'].append([str(src), str(dst)])
        for src, dst in copies:
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            log['copies'].append(str(dst))
    finally:
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"undo log: {log_path}")
    if OLD.is_dir() and not (OLD / '.gitignore').exists():
        (OLD / '.gitignore').write_text('*\n', encoding='utf-8')
    for d in DS.glob('*') if DS.is_dir() else []:
        n = len(list((d / 'images').glob('*'))) if (d / 'images').is_dir() else 0
        (d / 'dataset.json').write_text(json.dumps({'name': d.name, 'images': n,
                                                    'created': log['created']}, indent=1), encoding='utf-8')


def undo(path):
    log = json.loads(Path(path).read_text(encoding='utf-8'))
    for dst in reversed(log['copies']):
        p = Path(dst)
        if p.is_file():
            p.unlink()
    for src, dst in reversed(log['moves']):
        if Path(dst).exists() and not Path(src).exists():
            Path(src).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(dst, src)
    # remove files this script wrote itself, then folders left empty by the undo
    for f in list(DS.glob('*/dataset.json')) + [OLD / '.gitignore']:
        if f.is_file():
            f.unlink()
    for root in (DS, OLD):
        if root.is_dir():
            for d in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: -len(p.parts)):
                try:
                    d.rmdir()
                except OSError:
                    pass
    for d in (DS, DS.parent, OLD):
        try:
            d.rmdir()
        except OSError:
            pass
    print(f"undone: {len(log['moves'])} moves, {len(log['copies'])} copies")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--undo', metavar='LOG')
    ap.add_argument('--no-legacy', action='store_true', help='do not build legacy_v0')
    ap.add_argument('--verbose', '-v', action='store_true', help='list every file')
    a = ap.parse_args()
    if a.undo:
        undo(a.undo)
        return
    if not TD.is_dir():
        sys.exit(f"{TD} not found")

    moves = plan_old_models()
    copies, notes = plan_odir_local()
    if not a.no_legacy:
        c2, n2 = plan_legacy()
        copies += c2; notes += n2

    size = sum(s.stat().st_size for s, _ in moves)
    print(f"1) MOVE {len(moves)} old model/run files ({human(size)}) -> {OLD}")
    groups = {}
    for s, _ in moves:
        groups.setdefault(str(s.parent.relative_to(TD)), []).append(s)
    for g, files in sorted(groups.items()):
        print(f"     training-data/{g}: {len(files)} file(s)")
        if a.verbose:
            for f in files:
                print(f"        {f.name}")
    print(f"2) COPY into {DS}:")
    for n in notes:
        print(f"     {n}")
    if a.verbose:
        for s, d in copies:
            print(f"        {s.relative_to(REPO)} -> {d.relative_to(REPO)}")
    if not a.apply:
        print("\nDry run - nothing changed. Run again with --apply.")
        return
    apply(moves, copies)
    print("done.")


if __name__ == '__main__':
    main()
