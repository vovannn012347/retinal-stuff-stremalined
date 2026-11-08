import os
import shutil
import csv

'''Move stuff to the learning folders with correct labels'''
source_correct_dir = "D:/Shared/Kursova/processing-scripts/retinal-glaucoma/retinal-stuff-streamlined/artifacts/retina_nerve_classifier/old/nerves_defined_output_correct"
source_incorrect_dir = "D:/Shared/Kursova/processing-scripts/retinal-glaucoma/retinal-stuff-streamlined/artifacts/retina_nerve_classifier/old/nerves_defined_output_iccorect"

dest_images_dir = "D:/Shared/Kursova/processing-scripts/retinal-glaucoma/retinal-stuff-streamlined/artifacts/retina_images_prepared/retina_classify/nerves_cropped_out"
dest_labels_dir = "D:/Shared/Kursova/processing-scripts/retinal-glaucoma/retinal-stuff-streamlined/artifacts/retina_images_prepared/retina_classify/nerves_cropped_out_labels"

labels_available = ['glaucoma', 'atrophy', 'valid_image']
correct_label = "valid_image"

label_files_dir = "D:/Shared/Kursova/processing-scripts/retinal-glaucoma/retinal-stuff-streamlined/artifacts/retina_nerve_classifier/old"
label_files_prefix = "label_variable_files_"
label_files_suffix = ".txt"

# ================================================


def ensure_dirs():
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)


def load_label_name_sets():
    """
    Loads:
      - label_variable_files_<label>.txt      -> full labels (1.0)
      - label_variable_files_<label>_half.txt -> suspected labels (0.5)

    Returns:
      {
        label: {
            "full": set(...),
            "half": set(...)
        }
      }
    """
    label_sets = {}

    for label in labels_available:
        full_set = set()
        half_set = set()

        full_fname = f"{label_files_dir}/{label_files_prefix}{label}{label_files_suffix}"
        half_fname = f"{label_files_dir}/{label_files_prefix}{label}_half{label_files_suffix}"

        if os.path.isfile(full_fname):
            with open(full_fname, "r", encoding="utf-8") as f:
                full_set = set(line.strip() for line in f if line.strip())

        if os.path.isfile(half_fname):
            with open(half_fname, "r", encoding="utf-8") as f:
                half_set = set(line.strip() for line in f if line.strip())

        label_sets[label] = {
            "full": full_set,
            "half": half_set,
        }

    return label_sets


def base_name_without_suffix(filename):
    """
    Removes extension and trailing _<number> if present.
    """
    name, _ = os.path.splitext(filename)
    if "_" in name and name.split("_")[-1].isdigit():
        return "_".join(name.split("_")[:-1])
    return name


def process_directory(src_dir, correctness_value, label_sets):
    """
    Copies images and creates per-image CSV label files.
    """
    for root, _, files in os.walk(src_dir):
        for fname in files:
            src_path = os.path.join(root, fname)

            if not os.path.isfile(src_path):
                continue

            # Copy image
            dst_img_path = os.path.join(dest_images_dir, fname)
            shutil.copy2(src_path, dst_img_path)

            base_name = base_name_without_suffix(fname)
            row = {}

            # correctness label
            for label in labels_available:
                if correctness_value == 0:
                    row[label] = 0.0
                else:
                    if base_name in label_sets[label]["full"]:
                        row[label] = 1.0
                    elif base_name in label_sets[label]["half"]:
                        row[label] = 0.5
                    else:
                        row[label] = 0.0

            row[correct_label] = float(correctness_value)

            # Write CSV
            label_fname = f"{fname}.csv"
            dst_label_path = os.path.join(dest_labels_dir, label_fname)

            with open(dst_label_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=labels_available
                )
                writer.writeheader()
                writer.writerow(row)


def main():
    ensure_dirs()
    label_sets = load_label_name_sets()

    process_directory(
        source_incorrect_dir,
        correctness_value=0.0,
        label_sets=label_sets
    )

    process_directory(
        source_correct_dir,
        correctness_value=1.0,
        label_sets=label_sets
    )


if __name__ == "__main__":
    main()