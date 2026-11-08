# SurgicalSlicing: тренування класифікатора диска зорового нерва з ітеративним вирізанням

Тут новий код. Старий (`code/nerve-classifier/16_01_2026`, ButcherSlicing) лишається як був.

Відмінність від ButcherSlicing. Там мережа спершу навмисно надлишкова, а потім ріжеться torch-pruning
по L1-магнітуді. Тут перед кожним вирізанням мережу готують: dropout поступово росте, канали,
які підуть під ніж, вибираються заздалегідь і поступово «гасяться» пеналізацією. До моменту
вирізання вони майже нульові, тому сам розріз майже нічого не змінює.

## 1. Сетап

### Ноут (Windows, CPU)
Використовується наявний `venv` у корені `retinal-stuff-streamlined`. Якщо його немає, скрипт
створить його з Python 3.11.
```bat
code_new\setup\setup_laptop_windows.bat
```
Скрипт ставить torch CPU, `requirements.txt` і запускає `check_env.py` (перевірка і замір швидкості).

### Ubuntu + NVIDIA GPU
Потрібен лише драйвер NVIDIA, системний CUDA toolkit не потрібен: CUDA runtime і cuDNN
приходять разом із колесом PyTorch.
```bash
sudo ubuntu-drivers install            # якщо драйвера ще немає, потім перезавантаження
bash code_new/setup/setup_ubuntu_cuda.sh
source .venv-cuda/bin/activate
```
Скрипт читає з `nvidia-smi` «CUDA Version» і за нею вибирає збірку torch:

| драйвер підтримує CUDA | збірка |
|---|---|
| 13.x і новіше | звичайний PyPI (зараз torch 2.15 + CUDA 13.2) |
| 12.8–12.9 | `whl/cu128` |
| 12.6–12.7 | `whl/cu126` |
| 11.8–12.5 | `whl/cu118` (старіший torch) |

Опції: `--venv <шлях>`, `--cpu`, `--python python3.12`. Кожен Linux venv окремий, Windows-ний `venv`
туди не підходить.

Перевірка будь-коли: `python code_new/setup/check_env.py [--require-cuda] [--vgg]`
(`--vgg` одразу качає ваги VGG16 для `model.init: vgg16`).

## 2. Дані

Тренер читає датасети такого вигляду:
```
training-data/classifier/datasets/<назва>/
    images/            кропи диска зорового нерва
    labels/            <ім'я файлу картинки>.csv  ->  glaucoma,atrophy,valid_image
```
Формат міток той самий, що пишуть старі prepare-скрипти. Мітка 0.5 («підозра») рахується як позитив.

Щоб розкласти `training-data/` під цей формат і прибрати старі моделі:
```bat
venv\Scripts\python.exe code_new\tools\organize_training_data.py            :: показує план, нічого не змінює
venv\Scripts\python.exe code_new\tools\organize_training_data.py --apply    :: виконує, пише undo-лог
venv\Scripts\python.exe code_new\tools\organize_training_data.py --undo training-data\organize_log_<час>.json
```
* старі моделі (`*.pth`, TensorBoard-логи, усе з папок `checkpoints`, логи `for-display`)
  **переносяться** в `old-training-data/` зі збереженням шляхів;
* датасети **копіюються** в `training-data/classifier/datasets/`:
  * `odir_local_v1` — набір, на якому вчився тренер 16_01_2026
    (`artifacts/retina_images_prepared/retina_classify/nerves_cropped_out` + мітки);
  * `legacy_v0` — старші кропи з `retina-stuff-classifier` плюс кропи дефінітора, зіставлені з мітками за іменем.
* решта (`retina-detection`, `retina-infill`, таблиці ODIR у `preprocess-output`) лишається на місці.

Повний ODIR-5K або інший набір кладеш так само (`images/` + `labels/`) і дописуєш у `datasets:` у конфігу.

## 3. Тренування

```bat
venv\Scripts\python.exe code_new\train_surgical.py --config code_new\configs\laptop.yaml
```
```bash
python code_new/train_surgical.py --config code_new/configs/gpu.yaml
python code_new/train_surgical.py --config code_new/configs/gpu.yaml --resume            # продовжити після зупинки
python code_new/train_surgical.py --config code_new/configs/gpu.yaml --set run.name=v2 --set surgical.cut_ratio=0.1
```
Будь-яке значення конфігу можна перекрити через `--set ключ=значення`. Ctrl+C зберігає стан;
продовження з `--resume`. Стан пишеться після кожного кроку по згортці.

Результати лежать у `training-runs/<run.name>/`. Папка сама себе ігнорує в git.

| файл | що це |
|---|---|
| `accepted/cycle_XX.pth` | модель, що пройшла поріг після XX вирізань (готова до використання) |
| `final_model.pth` | остання прийнята модель |
| `log.csv` / `steps.csv` | метрики по проходах / по кроках |
| `events.log` | увесь вивід |
| `split_<датасет>.json` | згортки по пацієнтах + відкладені тестові пацієнти |
| `state_last.pth` | точка продовження |

Графік: `python code_new/plot_log.py training-runs/<run.name>` → `log.png`.
Оцінка або передбачення:
`python code_new/evaluate.py --model .../final_model.pth --dataset <папка> [--split .../split_<назва>.json]`,
або `--images a.jpg b.jpg`.

## 4. Як працює метод

### Згортки (fold rotation)
* Датасет ділиться на `folds.k` згорток **по пацієнтах**: обидва ока й усі кропи одного пацієнта
  завжди в одній згортці. ODIR `1138_left_0.jpg` → пацієнт `1138`; локальні `028ІГБ02_1_0.jpg` → `028ІГБ`.
* Крок: вчимо на згортці `order[j]`, тестуємо на **наступній** `order[j+1]`, яку мережа в цьому
  проході ще не бачила.
* **Прохід** = кожна згортка натренована один раз. Після проходу порядок згорток перемішується.
* Кожні `folds.passes_per_dataset` проходів (за замовчуванням 3) скрипт переходить на наступний
  датасет зі списку (по колу).
* `folds.test_fraction` (10%) пацієнтів відкладається повністю, на них мережа не вчиться ніколи.
  Це чесна оцінка: валідація на «наступній згортці» з часом стає оптимістичною, бо згортки ротуються.

Розклади dropout і пеналізації та поріг якості рахуються **по проходах** (прохід = «епоха» цього методу).

### Цикл SurgicalSlicing
1. **Навчання** з `learn_dropout` (0), поки на зібраних за прохід передбаченнях «наступних згорток» не буде
   accuracy ≥ 0.90 і recall > 0.95 (recall = мінімум по `glaucoma`, `atrophy`).
   Тоді модель **приймається** (`accepted/cycle_XX.pth`).
2. **Dropout** +0.02 за прохід до 0.5, лише на шарах `surgical.dropout_layers`
   (кінці блоків encoder1..4, intermediate, fc).
3. Коли dropout доходить до **0.2**, вибираються канали під вирізання: `cut_ratio` (12.5%) найменш
   важливих за L1 у кожному шарі, плюс нейрони прихованого fc-шару. Відтоді раз за прохід:
   `w_вибраних *= (1 − λ)`, де λ росте на 6.6% від 0.5 (+0.033) за прохід і доходить до 0.5 разом із dropout.
   Альтернатива — `penalty_mode: l2` (штраф у лосі на кожному кроці).
4. Коли dropout = 0.5 і λ = 0.5, вибрані канали **вирізаються фізично**: нова, менша модель, ваги
   скопійовані точно. Оптимізатор створюється заново, lr × `lr_cycle_decay`, і цикл іде знову з кроку 1.
5. Якщо за `max_recover_passes` після вирізання поріг не досягнуто, скрипт повертає останню прийняту
   модель і повторює цикл із удвічі меншим `cut_ratio` (до `max_retries` разів).
6. Зупинка: `max_cycles` вирізань, або розмір ≤ `target_param_ratio` від початкового, або закінчились спроби.

Якщо під час рампи метрики падають нижче порогу більше ніж на `degrade_tolerance`, рампа ставиться на паузу
(до `max_hold_passes` проходів за цикл). Довше навчання: `max_learn_passes` / `max_recover_passes`.

### Модель
Та сама архітектура, що в 16_01_2026 (VGG16-подібна, ті самі імена шарів), тому старі `.pth`
вантажаться через `model.init: checkpoint`. Що змінено:
* ширина кожного шару задається явно, тож обрізану модель можна відтворити;
* вихід — логіти, лосс `BCEWithLogitsLoss`. Старий код рахував `CrossEntropyLoss` поверх Sigmoid,
  а для multi-label це неправильно. `pos_weight` штовхає recall вгору.

Препроцесинг кропу (`preprocess`): `size`; `mode: stretch | pad | center_crop`; `crop_fraction`.
Поки не вирішено, як найкраще вирізати й стискати диск, це можна міняти без правки коду.

## 5. Тести
```bash
python code_new/tests/test_surgery.py     # розріз точний, старі чекпойнти, тайминг розкладу, згортки, групування
python code_new/tests/smoke_test.py       # повний цикл на синтетичних даних з маленькою моделлю (кілька хвилин на CPU)
```

## 6. Далі
Скрипт для ProcessorApplication (PyTorch): бере `final_model.pth` (формат `surgical-slicing-v1`:
`arch` + `model` + `label_order` + `preprocess`), будує `GlaucomaNet(**arch)` і рахує ймовірності.
