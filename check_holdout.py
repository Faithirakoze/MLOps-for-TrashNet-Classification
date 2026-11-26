from pathlib import Path

HOLDOUT_DIR = Path("data/holdout")

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

total = 0

print("HOLDOUT DATASET CHECK")
print("---------------------")

for category in CLASSES:

    category_dir = HOLDOUT_DIR / category

    if not category_dir.exists():
        print(f"{category}: MISSING FOLDER")
        continue

    images = [
        file for file in category_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    count = len(images)
    total += count

    print(f"{category}: {count} images")

print("---------------------")
print(f"Total images: {total}")