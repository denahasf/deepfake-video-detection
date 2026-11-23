import json
import os
import shutil

dataset_dir = "dataset/dfdc_train_part_0"

with open(os.path.join(dataset_dir, "metadata.json")) as f:
    metadata = json.load(f)

for video, info in metadata.items():
    label = info["label"]

    src = os.path.join(dataset_dir, video)

    if label == "REAL":
        dst = os.path.join(dataset_dir, "real", video)
    else:
        dst = os.path.join(dataset_dir, "fake", video)

    if os.path.exists(src):
        shutil.move(src, dst)

print("Dataset sorted successfully!")