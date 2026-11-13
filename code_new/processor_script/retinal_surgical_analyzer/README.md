# glaucoma_retinal_analyzer_surgical (ProcessorApplication script package)

Fundus photos -> optic disc candidates (SurgicalSlicing definitor) -> crops from the original photo -> classifier
(glaucoma, atrophy, valid_image). The candidate with the highest valid_image is the disc; a photo with no accepted
candidate is reported as "retake".

## Ready to upload

1. Put the trained models into `weights/` (fixed names):
   - `definitor.pth` <- e.g. `training-runs/definitor_cpu_from_x16/final_model.pth`
   - `classifier.pth` <- e.g. `training-runs/classifier_laptop/final_model.pth`
2. Zip the CONTENTS of this folder (`script_manifest.json` at the archive root), or let the tool do both steps:
   ```
   .venv\Scripts\python.exe code_new\tools\build_processor_script.py --definitor <definitor.pth> --classifier <classifier.pth> --zip dist\retinal_surgical_analyzer.zip
   ```
3. Upload the zip on the provisioning page.

`code/rt_*.py` are generated from `code_new/surgical/` by the same tool (flat modules: the executor imports every
`script_file` as a top-level module). Edit the sources there and re-run the tool, not the copies.

## Inputs / outputs

| Input | Type | Default | Meaning |
|---|---|---|---|
| `retinal_images` | multi_file | - | photos as taken (same label as the old analyzer, so uploads are shared) |
| `positive_threshold` | decimal | 0.5 | glaucoma / atrophy reported as found from this probability |
| `suspect_threshold` | decimal | 0.3 | reported as suspected from this probability |
| `valid_threshold` | decimal | 0.5 | a candidate is accepted as the disc from this valid_image |
| `max_candidates` | integer | 3 | candidates checked per photo (largest first) |

| Output | Type | Meaning |
|---|---|---|
| `retinal_masks_images` | multi_file | every candidate crop (`retinal_discs/<n>_<photo>_<k>.png`) |
| `retinal_classify_result` | file | `retinal_result.json`: per photo status (ok / retake / error), chosen disc, scores, grades, all candidates with their scores, crop status, timing; model sizes and thresholds |
| `retinal_glaucoma_count`, `retinal_suspected_glaucoma_count`, `retinal_atrophy_count` | integer | photos per finding |
| `retinal_glaucoma_probability_0_1`, `retinal_atrophy_probability_0_1` | decimal | highest probability in the batch |
| `retinal_retake_count` | integer | photos where the disc was not found or every candidate was rejected |

Requirements (executor): torch 2.3, numpy 1.26, opencv-python 4.9, Pillow - all already in `executor.spec`.
