from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import unicodedata

def convert_unicode_fraction(text: str) -> str:
    # NFKC normalization breaks complex glyphs down into standard characters
    normalized = unicodedata.normalize('NFKC', text)
    
    # Replace the Unicode fraction slash (U+2044) with a standard slash
    return normalized.replace('⁄', '/')

def load_dataset(root_dir):
    root = Path(root_dir)
    cases = []

    if not root.exists():
        raise FileNotFoundError(
            f"Test directory not found: {root}"
        )

    for difficulty_dir in sorted(root.iterdir()):
        if not difficulty_dir.is_dir():
            continue

        difficulty = difficulty_dir.name

        for recipe_dir in sorted(difficulty_dir.iterdir()):
  
            if not recipe_dir.is_dir():
                continue

            print(f"scanning {recipe_dir}")

            recipe = recipe_dir.name

            gt_files = list(recipe_dir.glob("*_gt.txt"))

            ground_truth = None

            if len(gt_files) != 1:
                # print("no ground truth, passing None")
                pass
            else:
                ground_truth = gt_files[0].read_text(
                    encoding="utf-8"
                ).strip()
                ground_truth = convert_unicode_fraction(ground_truth)


            images = sorted(
                file for file in recipe_dir.iterdir()
                if file.is_file()
                and file.suffix.lower() == ".jpg"
            )

            if not images:
                print(f"No images found in {recipe_dir}")
                continue

            for image in images:
                variant = image.stem

                prefix = recipe.replace("recipe", "r")

                if variant.startswith(prefix + "_"):
                    variant = variant[len(prefix)+1:]

                img = cv2.imread(image)
                img_data = np.asarray(img)

                # append test case
                cases.append({
                    "image": img_data,
                    "ground_truth": ground_truth,
                    "difficulty": difficulty,
                    "label": recipe,
                    "variant": variant
                })
    return cases

