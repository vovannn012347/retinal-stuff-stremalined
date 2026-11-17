"""Частка влучань: чи містить хоча б одна вирізка знімка диск зорового нерва.

Вирізки:  <crops_dir>/<photo>_<k>.png          (напр. 0_right_0.png)
Лейбли:   <labels_dir>/<photo>_<k>.png.csv      (glaucoma,atrophy,valid_image)
  valid_image == 1  -> диск видно повністю (влучання)
  0 < valid_image < 0.5 -> диск видно неповністю (часткове)
  valid_image == 0 або немає .csv -> помилкова вирізка, диска немає (промах)

Знімок без жодної вирізки береться з benchmark_per_photo.csv (n_islands == 0).

Запуск:
  python hit_rate.py --crops <папка вирізок> --labels <папка лейблів> \
                     --per-photo benchmark_per_photo.csv [--out hit_rate_per_photo.csv]
"""
import argparse, csv, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--crops", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--per-photo", help="benchmark_per_photo.csv: щоб урахувати знімки без кандидата")
ap.add_argument("--out", default="hit_rate_per_photo.csv")
a = ap.parse_args()

IMG = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
name_re = re.compile(r"^(.*)_(\d+)$")

def read_valid(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = {k.strip().lower(): v for k, v in rows[0].items()}
    try:
        return float(r.get("valid_image", "0"))
    except ValueError:
        return 0.0

photos = defaultdict(lambda: {"crops": 0, "valid": 0, "partial": 0, "missing": 0})
for fn in sorted(os.listdir(a.crops)):
    stem, ext = os.path.splitext(fn)
    if ext.lower() not in IMG:
        continue
    m = name_re.match(stem)
    photo = m.group(1) if m else stem
    p = photos[photo]
    p["crops"] += 1
    lab = os.path.join(a.labels, fn + ".csv")
    if not os.path.exists(lab):
        p["missing"] += 1
    else:
        v = read_valid(lab)
        if v is None or v == 0:      # valid_image = 0: not the disc -> miss
            p["missing"] += 1
        elif v >= 0.5:               # whole (or nearly whole) disc -> hit
            p["valid"] += 1
        else:                        # disc partly visible
            p["partial"] += 1

no_cand = []
if a.per_photo:
    with open(a.per_photo, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stem = os.path.splitext(r["photo"])[0]
            if int(float(r.get("n_islands", 0) or 0)) == 0 and stem not in photos:
                no_cand.append(stem)

hit = [k for k, p in photos.items() if p["valid"] > 0]
partial = [k for k, p in photos.items() if p["valid"] == 0 and p["partial"] > 0]
miss = [k for k, p in photos.items() if p["valid"] == 0 and p["partial"] == 0]
total = len(photos) + len(no_cand)

def pct(n, d):
    return f"{n} ({100 * n / d:.1f} %)" if d else str(n)

print(f"Знімків усього:                   {total}")
print(f"  з кандидатом:                   {len(photos)}")
print(f"  без кандидата:                  {pct(len(no_cand), total)}")
print(f"Влучання (є вирізка з повним диском): {pct(len(hit), total)}   серед знімків з кандидатом: {pct(len(hit), len(photos))}")
print(f"Часткові (диск видно неповністю): {pct(len(partial), total)}")
print(f"Промах (вирізки без диска):       {pct(len(miss), total)}")
nc = sum(p["crops"] for p in photos.values())
print(f"Вирізок усього: {nc}; повний диск {sum(p['valid'] for p in photos.values())}, "
      f"неповний {sum(p['partial'] for p in photos.values())}, без лейбла {sum(p['missing'] for p in photos.values())}")
multi = [k for k, p in photos.items() if p["crops"] > 1]
print(f"Знімків з кількома вирізками: {len(multi)}; з них влучань: {sum(1 for k in multi if photos[k]['valid'] > 0)}")

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["photo", "crops", "valid", "partial", "no_label", "result"])
    for k in sorted(photos):
        p = photos[k]
        res = "hit" if p["valid"] else ("partial" if p["partial"] else "miss")
        w.writerow([k, p["crops"], p["valid"], p["partial"], p["missing"], res])
    for k in sorted(no_cand):
        w.writerow([k, 0, 0, 0, 0, "no_candidate"])
print(f"Подробиці по знімках: {a.out}")
