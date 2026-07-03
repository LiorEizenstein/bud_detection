# Bud Detector

A YOLOv8-based object detection model trained to detect buds in vineyard images.

## Files

| File | Description |
|---|---|
| `prepare_dataset.py` | Converts labels and splits dataset into 80/20 train/val |
| `train.py` | Trains YOLOv8n on the prepared dataset |
| `inference.py` | Runs the trained model over a directory of images |
| `best.pt` | Trained model weights (best checkpoint, epoch 62) |
| `Report.pdf` | Full training report with methodology, results, and analysis |

## Usage

### 1. Prepare the dataset
```bash
python prepare_dataset.py
```

### 2. Train
```bash
python train.py
```

### 3. Run inference
```bash
python inference.py --source path/to/images --weights best.pt
```

## Results

- **mAP@50:** 0.301
- **Precision:** 0.497
- **Recall:** 0.324
- Training stopped at epoch 82 via early stopping (best at epoch 62)
