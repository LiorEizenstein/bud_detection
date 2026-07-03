import os
import random
import shutil

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'treex-dataset')
IMAGES_SRC = os.path.join(DATASET_DIR, 'images')
LABELS_SRC = os.path.join(DATASET_DIR, 'labels')
SPLIT_DIRS = {
    'train': (os.path.join(DATASET_DIR, 'images', 'train'),
              os.path.join(DATASET_DIR, 'labels', 'train')),
    'val':   (os.path.join(DATASET_DIR, 'images', 'val'),
              os.path.join(DATASET_DIR, 'labels', 'val')),
}
DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')


def convert_label_file(src_path, dst_path):
    with open(src_path, 'r') as f:
        lines = f.readlines()
    converted = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        x1, y1 = float(parts[1]), float(parts[2])
        x3, y3 = float(parts[5]), float(parts[6])
        cx = (x1 + x3) / 2
        cy = (y1 + y3) / 2
        w  = abs(x3 - x1)
        h  = abs(y3 - y1)
        converted.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    with open(dst_path, 'w') as f:
        f.writelines(converted)


def main():
    stems = []
    for fname in os.listdir(IMAGES_SRC):
        if not fname.lower().endswith('.jpg'):
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(LABELS_SRC, stem + '.txt')
        if os.path.exists(label_path):
            stems.append(stem)

    random.seed(42)
    random.shuffle(stems)
    split = int(len(stems) * 0.8)
    splits = {'train': stems[:split], 'val': stems[split:]}

    for split_name, (img_dir, lbl_dir) in SPLIT_DIRS.items():
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for stem in splits[split_name]:
            shutil.copy(
                os.path.join(IMAGES_SRC, stem + '.jpg'),
                os.path.join(img_dir, stem + '.jpg')
            )
            convert_label_file(
                os.path.join(LABELS_SRC, stem + '.txt'),
                os.path.join(lbl_dir, stem + '.txt')
            )

    abs_dataset = os.path.abspath(DATASET_DIR).replace('\\', '/')
    with open(DATA_YAML, 'w') as f:
        f.write(f"path: {abs_dataset}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['bud']\n")

    print(f"Train: {len(splits['train'])} images")
    print(f"Val:   {len(splits['val'])} images")
    print(f"data.yaml written to {DATA_YAML}")


if __name__ == '__main__':
    main()
