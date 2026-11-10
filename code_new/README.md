# SurgicalSlicing: тренування двох кроків з ітеративним вирізанням

Два кроки, один тренер (`train_surgical.py`); крок задається ключем `task` у конфігу:

| крок | task | що робить | конфіги |
|---|---|---|---|
| 1 | `definitor` | знаходить диск зорового нерва: знімок очного дна → маска диска | `configs/definitor_*.yaml` |
| 2 | `classifier` | аналізує кроп диска: glaucoma / atrophy / valid_image | `configs/classifier_*.yaml` |

Відмінність від ButcherSlicing. Там мережа спершу навмисно надлишкова, а потім ріжеться torch-pruning.
Тут перед кожним вирізанням мережу готують: dropout поступово росте, канали під ніж вибираються
заздалегідь і поступово «гасяться» пеналізацією. До моменту вирізання вони майже нульові, тому сам
розріз майже нічого не змінює.

## 1. Сетап

**Ноут (Windows, CPU):** `code_new\setup\setup_laptop_windows.bat`. Ставить torch CPU і `requirements.txt`
у venv у корені репозиторію, потім запускає `check_env.py`.

**Ubuntu + NVIDIA GPU:** потрібен лише драйвер (CUDA і cuDNN приходять разом із колесом PyTorch).
```bash
sudo ubuntu-drivers install            # якщо драйвера ще немає, потім перезавантаження
bash code_new/setup/setup_ubuntu_cuda.sh
source .venv-cuda/bin/activate
```
Скрипт вибирає збірку torch за «CUDA Version» з `nvidia-smi`:

| драйвер підтримує | збірка |
|---|---|
| CUDA 13+ | звичайний PyPI |
| CUDA 12.8 | `cu128` |
| CUDA 12.6 | `cu126` |
| CUDA 11.8 | `cu118` |

Опції: `--venv <шлях>`, `--cpu`, `--python python3.12`.
Перевірка будь-коли: `python code_new/setup/check_env.py [--require-cuda] [--vgg]`.

## 2. Дані

**Definitor** — пари картинка + маска, зіставлені за іменем:
```
training-data/source-input/input_for_definitor/015МЛФ01.jpg
training-data/eye-nerve-definitor/training-masks/015МЛФ01.mask.jpg     (будь-яке розширення з обох боків)
```
Картинки без маски і маски без картинки пропускаються. Їхня кількість пишеться в лог.

Перед подачею в мережу:
* чорна рамка обрізається, маска обрізається тим самим прямокутником (`crop_to_retina`);
* картинка переводиться в сірий, як у старому тренері (`grayscale`);
* стискається до `size`×`size` (576 на GPU, 288 на ноуті; має ділитися на 16).

Аугментація (віддзеркалення, поворот) іде разом для картинки й маски.

**Classifier** — папка з `images/` і `labels/`, мітки у файлах `<ім'я картинки>.csv` з колонками
`glaucoma,atrophy,valid_image`. Мітка 0.5 («підозра») рахується як позитив.

Кожен датасет у списку `datasets:` описується так само. Тренер проходить їх по колу: кожні
`folds.passes_per_dataset` проходів бере наступний.

## 3. Тренування

```bash
python code_new/train_surgical.py --config code_new/configs/definitor_gpu.yaml
python code_new/train_surgical.py --config code_new/configs/definitor_gpu.yaml --resume       # продовжити
python code_new/train_surgical.py --config code_new/configs/classifier_gpu.yaml --set run.name=v2 --set surgical.cut_ratio=0.1
```
На ноуті ті самі команди з `*_laptop.yaml`. Будь-яке значення конфігу можна перекрити через `--set ключ=значення`.
Ctrl+C зберігає стан; продовження з `--resume`.

Результати лежать у `training-runs/<run.name>/`. Папка сама себе ігнорує в git.

| файл | що це |
|---|---|
| `accepted/cycle_XX.pth` | модель, що пройшла поріг після XX вирізань |
| `final_model.pth` | остання прийнята модель |
| `log.csv` / `steps.csv` | метрики по проходах / по кроках |
| `mistakes.csv` | семпли, впорядковані за помилкою (їх повторює mistake replay) |
| `events.log` | увесь вивід |
| `split_<датасет>.json` | згортки по пацієнтах + відкладені тестові пацієнти |
| `state_last.pth` | точка продовження |

Графік: `python code_new/plot_log.py training-runs/<run.name>` → `log.png`.

Оцінка:
```bash
python code_new/evaluate.py --config <конфіг> --model .../final_model.pth [--split .../split_<датасет>.json]
python code_new/evaluate.py --config <конфіг> --model .../final_model.pth --images a.jpg b.jpg --out masks/
```
Для definitor з `--images` пише маски в `--out`; для classifier друкує ймовірності.

## 4. Як працює

### Згортки
* Датасет ділиться на `folds.k` згорток **по пацієнтах**:
  * ODIR `1138_left_0` → `1138`;
  * локальні `027ІГБ01` / `028ІГБ02` → `ІГБ` (номер змінюється на кожне око, тому ключ — ініціали);
  * інші імена (`eye_case_3`) — окремий пацієнт на файл.
* Крок: вчимо на згортці `order[j]`, перевіряємо на наступній `order[j+1]`. **Прохід** = кожна згортка
  один раз; після проходу порядок перемішується.
* `folds.test_fraction` пацієнтів не бере участі в навчанні взагалі: це чесна оцінка.

### Mistake replay
Помилку кожного семпла (0…1) міряють, коли його згортка є «наступною» (чиста картинка, до навчання на
ній у цьому проході), і зберігають як ковзне середнє.
* **definitor:** 1 − Tversky-оцінка картинки (пропущений піксель диска важить більше за зайвий);
* **classifier:** найгірша мітка, |ймовірність − ціль| (≥ 0.5 = мітка неправильна).

Коли згортка йде в навчання, семпли з помилкою ≥ `min_error` повторюються:
`додаткові копії = min(max_extra, floor(gain · помилка / min_error))`. З `min_error` 0.25 це дає
0.25 → +1, 0.5 → +2, ≥ 0.75 → +3. Список найгірших пишеться в `mistakes.csv`.

### Метрики й поріг
* **definitor** (пікселі, зведені за прохід):
  * recall = знайдені пікселі диска / усі пікселі диска (головне);
  * precision = знайдені пікселі диска / усі передбачені пікселі.

  Поріг: recall ≥ 0.95 і precision ≥ 0.70. Лосс: BCE (`pos_weight` 3) + Tversky (β 0.7 > α 0.3),
  тобто пропуск штрафується сильніше. Додатково логуються IoU, Dice і recall найгіршої картинки.
* **classifier:** accuracy ≥ 0.90 (середня по мітках) і recall > 0.95 (мінімум по glaucoma, atrophy).

Поріг має триматися `gate_consecutive_passes` (2) проходи підряд.

### Цикл SurgicalSlicing
1. **Навчання** без dropout до порогу → модель **приймається** (`accepted/cycle_XX.pth`).
2. **Dropout** +0.02 за прохід до 0.5. Діє на dropout у кінці кожного блоку (encoder1..4, intermediate,
   decoder4..2 / fc).
3. Коли dropout доходить до **0.2**, у кожному шарі вибираються `cut_ratio` (12.5%) найслабших каналів.
   Раз за прохід їхні ваги множаться на (1 − λ); λ росте на 0.033 за прохід і доходить до 0.5 разом із dropout.
4. Коли dropout = 0.5 і λ = 0.5, канали **вирізаються фізично**. Для definitor це враховує skip-з'єднання:
   вирізаний канал енкодера зникає і з входу відповідного декодера. Далі цикл іде знову з кроку 1.
5. Якщо після вирізання поріг не досягнуто за `max_recover_passes`, скрипт повертає останню прийняту модель
   і пробує вирізати вдвічі менше. Зупинка: `max_cycles`, `target_param_ratio` або закінчились спроби.

### Моделі
* `DefinitorNet` — топологія `FcnskipNerveDefinitor2`. Старі чекпойнти дефінітора (з dropout і без)
  вантажаться через `model.init: checkpoint`.
* `GlaucomaNet` — топологія `HandmadeGlaucomaClassifier`, старі чекпойнти теж вантажаться.
* Обидві повертають логіти; ширина кожного шару зберігається у файлі моделі.

## 5. Тести
```bash
python code_new/tests/test_surgery.py     # розріз точний для обох мереж, старі чекпойнти, розклад, згортки, replay
python code_new/tests/smoke_test.py       # повний цикл для обох кроків на синтетичних даних (~1 хв на CPU)
```
