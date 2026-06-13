
from ultralytics import YOLO
from pathlib import Path
import yaml
import os


def validate_dataset_balance(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    train_images = Path(config['path']) / config['train']

    f_count, t_count = 0, 0

    for img in train_images.glob("*.jpg"):
        name = img.stem.lower()
        if "_f_" in name or "front" in name:
            f_count += 1
        elif "_t_" in name or "rear" in name or "back" in name:
            t_count += 1

    total = f_count + t_count

    if total == 0:
        print("Dataset is empty or no valid images found.")
        return

    print("Dataset balance:")
    print("Front:", f_count, "Rear:", t_count)


def main():
    DATA_YAML = "train.yaml"

    if not os.path.exists(DATA_YAML):
        print("YAML not found")
        return

    validate_dataset_balance(DATA_YAML)

    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=800,
        batch=16,
        device=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        name="multicamera_yolo11n_augmentation",
        save_period=10
    )


if __name__ == "__main__":
    main()
