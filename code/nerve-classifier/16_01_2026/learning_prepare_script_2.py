import os
import shutil
import csv

'''Move stuff to the '''
import os
import csv

DIRECTORY = "D:/Shared/Kursova/processing-scripts/retinal-glaucoma/retinal-stuff-streamlined/artifacts/retina_images_prepared/retina_classify/nerves_cropped_out_labels"  # change this

for filename in os.listdir(DIRECTORY):
    if not filename.lower().endswith(".csv"):
        continue

    file_path = os.path.join(DIRECTORY, filename)

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Skip empty files
    if not rows:
        continue

    row = rows[0]

    glaucoma = float(row["glaucoma"])
    atrophy = float(row["atrophy"])

    if atrophy == 0.0 and glaucoma != 0.0:
        row["atrophy"] = str(glaucoma)

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
