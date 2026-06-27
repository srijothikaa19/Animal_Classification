from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

SELECTED = Path("data/selected")

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

OUTPUT_DIRS = [
    Path("data/train"),
    Path("data/val"),
    Path("data/test")
]

# clean old split folders
for output_dir in OUTPUT_DIRS:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

classes = sorted([folder.name for folder in SELECTED.iterdir() if folder.is_dir()])

for class_name in classes:
    images = list((SELECTED / class_name).glob("*"))

    train_files, temp_files = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=42
    )

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, files in split_map.items():
        split_class_dir = Path("data") / split_name / class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            shutil.copy2(file, split_class_dir / file.name)

    print(
        f"{class_name}: "
        f"train={len(train_files)}, "
        f"val={len(val_files)}, "
        f"test={len(test_files)}"
    )

print("\nDataset split complete.")