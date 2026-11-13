"""Pick a balanced training subset of a classifier dataset; the rest becomes the benchmark set.

    python code_new/tools/make_train_subset.py training-data/eye-classifier/odir_local_x9_v3 --n 500

Reads <dataset>/labels/ (only crops that also exist in images/), puts every crop in one category
    glaucoma   glaucoma > 0              (incl. 0.5 "suspected")
    atrophy    atrophy > 0 (and glaucoma = 0)
    invalid    valid_image = 0           (not the disc)
    partial    0 < valid_image < 1
    normal     everything else (valid_image = 1, glaucoma = atrophy = 0)
and picks about --n crops with the shares below (a category with fewer crops gives all it has, the rest
of its share goes to the others). Selection is by PATIENT: all crops of a picked patient go to training
(both eyes, every island), so no patient is in training and benchmark at the same time.

Writes  <dataset>/labels_train/   label files of the training subset  (classifier_*.yaml reads these)
        <dataset>/labels_bench/   label files of everything else     (classifier_bench.yaml evaluates these)
        <dataset>/subset.csv      crop, patient, category, set
Both folders are rebuilt on every run; labels/ and images/ are not touched.

Options: --n 500 --seed 7 --shares glaucoma=0.3,atrophy=0.25,invalid=0.15,partial=0.1,normal=0.2
         --max-train-frac 0.7 (at most 70 % of any category in training, so the benchmark has every kind)
"""
import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgical.data import group_key                                  # noqa: E402

ORDER = ('glaucoma', 'atrophy', 'valid_image')
CATS = ('glaucoma', 'atrophy', 'invalid', 'partial', 'normal')
DEFAULT_SHARES = 'glaucoma=0.3,atrophy=0.25,invalid=0.15,partial=0.1,normal=0.2'


def read(f: Path):
    try:
        with open(f, encoding='utf-8-sig', newline='') as fh:
            row = next(csv.DictReader(fh))
        return {k: float((row.get(k) or '0').strip().replace(',', '.')) for k in ORDER}
    except Exception:                                             # noqa: BLE001
        return None


def category(v):
    if v['glaucoma'] > 0:
        return 'glaucoma'
    if v['atrophy'] > 0:
        return 'atrophy'
    if v['valid_image'] == 0:
        return 'invalid'
    if v['valid_image'] < 1:
        return 'partial'
    return 'normal'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--n', type=int, default=500, help='about this many crops for training')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--shares', default=DEFAULT_SHARES)
    ap.add_argument('--max-train-frac', type=float, default=0.7,
                    help='at most this fraction of any category goes to training (the benchmark keeps the rest)')
    a = ap.parse_args()

    ds = Path(a.dataset)
    labels, images = ds / 'labels', ds / 'images'
    shares = {k: float(v) for k, v in (p.split('=') for p in a.shares.split(','))}
    if set(shares) - set(CATS):
        sys.exit(f"unknown category in --shares: {set(shares) - set(CATS)}")
    tot = sum(shares.values())
    shares = {c: shares.get(c, 0.0) / tot for c in CATS}

    crops = {}                                                    # label file name -> (patient, category)
    bad = 0
    for f in sorted(labels.glob('*.csv')):
        if not (images / f.name[:-4]).is_file():
            continue
        v = read(f)
        if v is None:
            bad += 1
            continue
        crops[f.name] = (group_key(f.name[:-4], None), category(v))
    if not crops:
        sys.exit(f"no labelled crops in {ds}")
    by_patient = defaultdict(list)
    for name, (p, c) in crops.items():
        by_patient[p].append(name)
    have = defaultdict(int)
    for _, c in crops.values():
        have[c] += 1

    # quotas: share x n, capped by what exists; leftovers go to the categories that still have crops
    cap = {c: int(have[c] * a.max_train_frac) for c in CATS}
    quota = {c: min(cap[c], round(shares[c] * a.n)) for c in CATS}
    left = a.n - sum(quota.values())
    while left > 0:
        open_ = [c for c in CATS if quota[c] < cap[c] and shares[c] > 0] or [c for c in CATS if quota[c] < cap[c]]
        if not open_:
            break
        for c in open_:
            if left <= 0:
                break
            quota[c] += 1
            left -= 1

    rng = random.Random(a.seed)
    pool = {c: [n for n, (_, cc) in crops.items() if cc == c] for c in CATS}
    for c in CATS:
        rng.shuffle(pool[c])
    taken_pat, train, got = set(), set(), defaultdict(int)
    # rare categories first, so their patients are not used up by the common ones
    for c in sorted(CATS, key=lambda c: have[c]):
        for name in pool[c]:
            if got[c] >= quota[c]:
                break
            p = crops[name][0]
            if p in taken_pat:
                continue
            members = by_patient[p]
            if len(train) + len(members) > a.n * 1.03:          # stay close to --n
                continue
            if any(got[crops[m][1]] + 1 > cap[crops[m][1]] for m in members):
                continue
            taken_pat.add(p)
            for m in by_patient[p]:                              # the whole patient goes to training
                train.add(m)
                got[crops[m][1]] += 1

    for sub in ('labels_train', 'labels_bench'):
        d = ds / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
    with open(ds / 'subset.csv', 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['crop', 'patient', 'category', 'set'])
        for name in sorted(crops):
            s = 'train' if name in train else 'bench'
            shutil.copy2(labels / name, ds / f'labels_{s}' / name)
            w.writerow([name[:-4], crops[name][0], crops[name][1], s])

    bench = {c: have[c] - sum(1 for n in train if crops[n][1] == c) for c in CATS}
    print(f"{len(crops)} labelled crops, {len(by_patient)} patients" + (f" ({bad} unreadable label files skipped)" if bad else ''))
    print(f"{'category':<10} {'all':>6} {'train':>6} {'bench':>6}")
    for c in CATS:
        tr = sum(1 for n in train if crops[n][1] == c)
        print(f"{c:<10} {have[c]:>6} {tr:>6} {bench[c]:>6}")
    print(f"{'total':<10} {len(crops):>6} {len(train):>6} {len(crops) - len(train):>6}   "
          f"({len(taken_pat)} patients in training, none of them in the benchmark)")
    print(f"-> {ds / 'labels_train'}  and  {ds / 'labels_bench'}")


if __name__ == '__main__':
    main()
