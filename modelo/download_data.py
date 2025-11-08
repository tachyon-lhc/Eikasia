import requests
import numpy as np
import os
import tempfile

CLASSES = [
    "cat",
    "dog",
    "fish",
    "bird",
    "eye",
    "face",
    "smiley face",
    "house",
    "tree",
    "flower",
    "sun",
    "moon",
    "star",
    "cloud",
    "car",
    "bicycle",
    "apple",
    "cup",
    "book",
    "clock",
]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DATASET_DIR = "modelo/dataset"
SAMPLES_PER_CLASS = 1000


def download_and_sample_class(class_name):
    """Descarga y guarda solo N muestras de una clase"""
    filepath = os.path.join(DATASET_DIR, f"{class_name}.npy")

    if os.path.exists(filepath):
        print(f"✓ {class_name} ya existe")
        return

    print(f"Descargando {class_name}...", end=" ", flush=True)

    try:
        url = f"{BASE_URL}{class_name}.npy"
        response = requests.get(url)
        response.raise_for_status()

        # Guardar en archivo temporal
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Cargar desde temporal
        data = np.load(tmp_path)

        # Eliminar temporal
        os.unlink(tmp_path)

        # Tomar solo las primeras N muestras
        sampled_data = data[:SAMPLES_PER_CLASS]

        # Guardar subset
        np.save(filepath, sampled_data)

        print(f"✓ ({len(sampled_data)} muestras)")

    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("=== Eikasia - Descarga de Dataset ===\n")
    print(f"Clases: {len(CLASSES)}")
    print(f"Muestras por clase: {SAMPLES_PER_CLASS}")
    print(f"Total de imágenes: {len(CLASSES) * SAMPLES_PER_CLASS:,}\n")

    for class_name in CLASSES:
        download_and_sample_class(class_name)

    print("\n=== Descarga completada ===")

    # Verificar
    total_size = 0
    print("\nArchivos guardados:")
    for class_name in CLASSES:
        filepath = os.path.join(DATASET_DIR, f"{class_name}.npy")
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            total_size += size_mb
            data = np.load(filepath)
            print(f"  {class_name:15} {len(data):,} ejemplos - {size_mb:.1f} MB")

    print(f"\n📦 Tamaño total: {total_size:.1f} MB")


if __name__ == "__main__":
    main()
