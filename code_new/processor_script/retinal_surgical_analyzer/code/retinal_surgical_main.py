"""ProcessorApplication entry point: fundus photos -> optic disc candidates -> glaucoma / atrophy.

Pipeline per photo (same steps as training, see rt_pipeline.py):
    original photo -> retina crop, HSV-S + CLAHE-LAB, 512 px -> definitor (SurgicalSlicing model)
    -> probability map -> every island (disc candidate; dim photos: relative threshold)
    -> crop of each island from the ORIGINAL photo (1.75 disc diameters, at least 0.20 x retina width)
    -> classifier -> glaucoma, atrophy, valid_image per crop
The crop with the highest valid_image is taken as the disc. A photo without islands, or whose candidates
are all rejected by the classifier (valid_image < threshold), is reported as "retake" - the disc was not
found or the photo cannot be graded.

process(input_map, output_map) -> (summary, direct_outputs, file_outputs), see the module contract.
"""
import json
import os
import time

import numpy as np
from PIL import Image

import rt_pipeline as P

LABELS = ('glaucoma', 'atrophy')


def _num(v, default):
    try:
        f = float(v)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    v = str(v)
    if os.path.isdir(v):
        return [os.path.join(v, f) for f in sorted(os.listdir(v))]
    return [v]


def _grade(p, positive, suspect):
    if p >= positive:
        return 'positive'
    if p >= suspect:
        return 'suspected'
    return 'negative'


def process(input_map, output_map):
    photos = [p for p in _as_list(input_map.get('retinal_images'))
              if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))]
    if not photos:
        raise ValueError('no retinal images were given')
    proc_root = input_map.get('__processing_dir__', os.getcwd())
    crops_dir = output_map.get('retinal_masks_images')
    result_json = output_map.get('retinal_classify_result')
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(os.path.dirname(result_json), exist_ok=True)

    positive = _num(input_map.get('positive_threshold'), 0.5)
    suspect = _num(input_map.get('suspect_threshold'), 0.3)
    valid_thr = _num(input_map.get('valid_threshold'), 0.5)
    max_islands = int(_num(input_map.get('max_candidates'), 3)) or 3

    t_load = time.perf_counter()
    definitor = P.DefinitorRunner(input_map.get('definitor_model'))
    classifier = P.ClassifierRunner(input_map.get('classifier_model'))
    load_s = time.perf_counter() - t_load

    summary = {'status': 'success', 'message': '', 'errors': []}
    results, crop_files = [], []
    counts = {'glaucoma': 0, 'suspected_glaucoma': 0, 'atrophy': 0, 'suspected_atrophy': 0,
              'normal': 0, 'retake': 0}
    max_p = {k: 0.0 for k in LABELS}
    times = []

    for idx, path in enumerate(photos):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        rec = {'image': name, 'status': 'ok', 'candidates': [], 'disc': None}
        t0 = time.perf_counter()
        try:
            rgb = np.asarray(Image.open(path).convert('RGB'))
            prob = definitor.probability(rgb)
            islands = P.find_islands(prob, definitor.threshold)[:max_islands]
            crops, metas = [], []
            for isl in islands:
                info, crop = P.crop_island(rgb, isl)
                if crop is None:
                    continue
                fname = f"{idx:04d}_{stem}_{isl.index}.png"
                fpath = os.path.join(crops_dir, fname)
                Image.fromarray(crop).save(fpath)
                rel = os.path.relpath(fpath, proc_root).replace('\\', '/')
                crop_files.append(rel)
                crops.append(crop)
                metas.append({'file': rel, 'island': isl.index, 'fallback': bool(isl.fallback),
                              'mean_prob': round(isl.mean_prob, 4), 'crop_status': info.status,
                              'completeness': round(float(info.completeness), 3),
                              'center': [round(float(info.center_x), 1), round(float(info.center_y), 1)],
                              'disc_diameter_px': round(float(info.diameter_px), 1),
                              'crop_enlarged': bool(info.enlarged)})
            scores = classifier.predict(crops)
            for m, s in zip(metas, scores):
                m['scores'] = {k: round(v, 4) for k, v in s.items()}
            rec['candidates'] = metas
            valid = [m for m in metas if m['scores'].get('valid_image', 1.0) >= valid_thr]
            if not metas:
                rec['status'], rec['reason'] = 'retake', 'optic disc not found in the photo'
            elif not valid:
                rec['status'], rec['reason'] = 'retake', 'no candidate was accepted as the optic disc'
            else:
                disc = max(valid, key=lambda m: m['scores'].get('valid_image', 0.0))
                rec['disc'] = disc['file']
                rec['scores'] = {k: disc['scores'].get(k, 0.0) for k in LABELS}
                rec['grades'] = {k: _grade(rec['scores'][k], positive, suspect) for k in LABELS}
                if disc['crop_status'] == 'partial':
                    rec['warning'] = 'optic disc is cut by the edge of the photo'
                for k in LABELS:
                    max_p[k] = max(max_p[k], rec['scores'][k])
                    if rec['grades'][k] == 'positive':
                        counts[k] += 1
                    elif rec['grades'][k] == 'suspected':
                        counts['suspected_' + k] += 1
                if all(g == 'negative' for g in rec['grades'].values()):
                    counts['normal'] += 1
            if rec['status'] == 'retake':
                counts['retake'] += 1
        except Exception as e:                      # one bad photo must not stop the batch
            rec['status'], rec['reason'] = 'error', str(e)
            summary['errors'].append({'file': name, 'error': str(e)})
        rec['seconds'] = round(time.perf_counter() - t0, 3)
        times.append(rec['seconds'])
        results.append(rec)

    report = {
        'pipeline': {'definitor': definitor.describe(), 'classifier': classifier.describe(),
                     'thresholds': {'positive': positive, 'suspected': suspect, 'valid_image': valid_thr},
                     'model_load_seconds': round(load_s, 3),
                     'median_seconds_per_photo': round(float(np.median(times)), 3) if times else None},
        'counts': counts,
        'images': results,
    }
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if summary['errors']:
        summary['status'] = 'warning'
    summary['message'] = (f"{len(photos)} photos: {counts['glaucoma']} glaucoma, {counts['suspected_glaucoma']} "
                          f"suspected glaucoma, {counts['atrophy']} atrophy, {counts['normal']} normal, "
                          f"{counts['retake']} to retake")
    direct_outputs = {
        'retinal_glaucoma_count': counts['glaucoma'],
        'retinal_glaucoma_probability_0_1': round(max_p['glaucoma'], 4),
        'retinal_suspected_glaucoma_count': counts['suspected_glaucoma'],
        'retinal_atrophy_count': counts['atrophy'],
        'retinal_atrophy_probability_0_1': round(max_p['atrophy'], 4),
        'retinal_retake_count': counts['retake'],
    }
    file_outputs = {
        'retinal_masks_images': crop_files,
        'retinal_classify_result': [os.path.relpath(result_json, proc_root).replace('\\', '/')],
    }
    return summary, direct_outputs, file_outputs
