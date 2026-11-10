"""Move everything that is not part of the two working steps into unrelated/ (folder structure kept).

The two steps:
  1. optic nerve disc definitor  (code/nerve-definitor, FcnskipNerveDefinitor2)
  2. glaucoma classifier          (code/nerve-classifier, code_new SurgicalSlicing)

Unrelated = the retina detection and inpainting (infill) stages and their data/weights.
Dry run by default; nothing changes until --apply. Every move is logged for --undo.

    python code_new/tools/separate_unrelated.py                 # show the plan
    python code_new/tools/separate_unrelated.py --apply         # move (writes unrelated/move_log_<time>.json)
    python code_new/tools/separate_unrelated.py --undo unrelated/move_log_<time>.json
    python code_new/tools/separate_unrelated.py --keep training-data/retina-detection    # keep a path in place

Files that match neither list are reported as "unclassified" and left where they are.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / 'unrelated'

# never looked at
SKIP_TOP = {'.git', '.venv', 'venv', '.idea', 'unrelated', '.vscode'}

# unrelated: retina detection + inpainting (paths relative to the repo root, '/' = whole folder)
MOVE = [
    'model/losses_inpaint.py',
    'model/networks_inpaint.py',
    'model/networks_inpaint_tf.py',
    'model/retina_network.py',            # retina detection network (not imported by definitor/classifier)
    'utils/edge_detection.py',            # used only by the inpainting code (kornia)
    'utils/misc_inpaint.py',
    'test_normal.bat',                    # inpainting test commands
    'test_retina.bat',
    'artifacts/pretrained/',              # DeepFill inpainting weights (places2 / celebahq)
    'artifacts/testing/',
    'training-data/retina-detection/',
    'training-data/retina-infill/',
    'old-training-data/retina-detection/',
    'old-training-data/retina-infill/',
]

# related: definitor + classifier (and what they import)
KEEP = [
    '.gitignore', 'info_changes.txt',
    'code/', 'code_new/', 'config/',
    'model/__init__.py',                  # package file (its inpainting loader is lazy, never called by the two steps)
    'model/retina_classifier_networks.py',
    'utils/__init__.py', 'utils/misc_retina.py', 'utils/retinaldata.py', 'utils/mics.py',
    'artifacts/raw_data/', 'artifacts/retina_images_prepared/',
    'artifacts/retina_nerve_classifier/', 'artifacts/retina_nerve_definitor/',
    'training-data/classifier/', 'training-data/eye-nerve-definitor/', 'training-data/retina-stuff-classifier/',
    'training-data/retina-stuff-definitor/', 'training-data/source-input/', 'training-data/organize_log_',
    'old-training-data/retina-stuff-definitor/', 'old-training-data/retina-stuff-classifier/',
    'old-training-data/.gitignore',
]

# what unrelated/ must not put into git (the moved code stays visible to git as a rename)
TARGET_GITIGNORE = "training-data/\nold-training-data/\nartifacts/\n*.pth\n__pycache__/\nmove_log_*.json\n"


def matches(rel: str, rules) -> bool:
    for r in rules:
        if r.endswith('/'):
            if rel.startswith(r):
                return True
        elif rel == r or (r.endswith('_') and rel.startswith(r)):
            return True
    return False


def moved_module_stems():
    return {Path(r).stem for r in MOVE if r.endswith('.py')}


def classify(keep_extra):
    keep = KEEP + [k.rstrip('/') + ('/' if (REPO / k).is_dir() else '') for k in keep_extra]
    stems = moved_module_stems()
    to_move, unclassified = [], []
    for p in sorted(REPO.rglob('*')):
        if not p.is_file():
            continue
        rel_path = p.relative_to(REPO)
        if rel_path.parts[0] in SKIP_TOP:
            continue
        rel = rel_path.as_posix()
        if matches(rel, keep):
            continue
        if matches(rel, MOVE):
            to_move.append(rel)
        elif p.parent.name == '__pycache__':
            # compiled copy of a moved module goes with it; other caches stay
            if p.name.split('.')[0] in stems:
                to_move.append(rel)
        else:
            unclassified.append(rel)
    return to_move, unclassified


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def summarize(rels):
    groups = {}
    for r in rels:
        parts = r.split('/')
        key = '/'.join(parts[:2]) + ('/' if len(parts) > 2 else '')
        g = groups.setdefault(key, [0, 0])
        g[0] += 1
        g[1] += (REPO / r).stat().st_size
    return groups


def apply(rels):
    TARGET.mkdir(exist_ok=True)
    gi = TARGET / '.gitignore'
    if not gi.exists():
        gi.write_text(TARGET_GITIGNORE, encoding='utf-8')
    log = {'created': datetime.now().isoformat(timespec='seconds'), 'moves': []}
    log_path = TARGET / f"move_log_{datetime.now():%Y%m%d_%H%M%S}.json"
    try:
        for rel in rels:
            src, dst = REPO / rel, TARGET / rel
            if dst.exists():
                print(f"  skip (already in unrelated/): {rel}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            log['moves'].append(rel)
    finally:
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"undo log: {log_path.relative_to(REPO)}")
    remove_empty_dirs({(REPO / r).parent for r in log['moves']}, REPO)


def remove_empty_dirs(dirs, stop):
    for d in sorted(dirs, key=lambda p: -len(p.parts)):
        while d != stop and d.is_dir():
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent


def undo(path):
    log = json.loads(Path(path).read_text(encoding='utf-8'))
    n = 0
    for rel in reversed(log['moves']):
        src, dst = REPO / rel, TARGET / rel
        if dst.exists() and not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
            n += 1
    remove_empty_dirs({(TARGET / r).parent for r in log['moves']}, TARGET)
    print(f"undone: {n} of {len(log['moves'])} files moved back")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--undo', metavar='LOG')
    ap.add_argument('--keep', action='append', default=[], metavar='PATH', help='keep this file/folder in place')
    ap.add_argument('--verbose', '-v', action='store_true', help='list every file')
    a = ap.parse_args()
    if a.undo:
        undo(a.undo)
        return

    to_move, unclassified = classify(a.keep)
    total = sum((REPO / r).stat().st_size for r in to_move)
    print(f"repo: {REPO}")
    print(f"MOVE -> unrelated/ : {len(to_move)} files, {human(total)}")
    for k, (n, s) in sorted(summarize(to_move).items()):
        print(f"   {k:55s} {n:6d} files  {human(s):>8s}")
    if a.verbose:
        for r in to_move:
            print(f"      {r}")
    if unclassified:
        print(f"\nUNCLASSIFIED (left in place, add to MOVE/KEEP in this script or use --keep): {len(unclassified)} files")
        for k, (n, s) in sorted(summarize(unclassified).items()):
            print(f"   {k:55s} {n:6d} files  {human(s):>8s}")
    if not a.apply:
        print("\nDry run - nothing changed. Run again with --apply.")
        return
    if not to_move:
        print("nothing to move")
        return
    apply(to_move)
    print("done.")


if __name__ == '__main__':
    sys.exit(main())
