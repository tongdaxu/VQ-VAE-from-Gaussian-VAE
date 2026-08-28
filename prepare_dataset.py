#!/usr/bin/env python3
from pathlib import Path
import argparse
from tqdm import tqdm

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root folder, e.g. /path/to/root")
    parser.add_argument(
        "-o", "--output",
        default="image_paths.txt",
        help="Output text file"
    )
    args = parser.parse_args()

    root = Path(args.root)

    if not root.is_dir():
        raise NotADirectoryError(f"Not a valid folder: {root}")

    class_dirs = [p for p in root.iterdir() if p.is_dir()]

    image_paths = []

    for class_dir in tqdm(class_dirs, desc="Scanning classes", unit="class"):
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(img_path.resolve())

    image_paths.sort()

    with open(args.output, "w", encoding="utf-8") as f:
        for img_path in tqdm(image_paths, desc="Writing paths", unit="image"):
            f.write(str(img_path) + "\n")

    print(f"Found {len(image_paths)} images.")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()