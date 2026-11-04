
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    "models/waste_classifier_v2.keras"
)

with open("models/class_names_v2.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

dataset_path = Path("dataset")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

correct = 0
total = 0

for folder in sorted(dataset_path.iterdir()):
    if not folder.is_dir():
        continue

    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"\nActual class: {folder.name}")

    if not images:
        print("No images found")
        continue

    class_correct = 0
    test_images = images[:10]

    for path in test_images:
        image = Image.open(path).convert("RGB").resize((224, 224))
        arr = np.array(image, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        scores = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(scores))
        predicted = class_names[idx]

        is_correct = predicted == folder.name
        total += 1
        correct += int(is_correct)
        class_correct += int(is_correct)

        print(
            f"{path.name} -> {predicted} "
            f"({scores[idx] * 100:.2f}%)"
        )

    print(f"Class result: {class_correct}/{len(test_images)} correct")

if total:
    print(f"\nOverall test result: {correct}/{total}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")
