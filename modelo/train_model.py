import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

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

DATASET_DIR = "modelo/dataset"
MODEL_PATH = "modelo/sketch_classifier.pkl"
ENCODER_PATH = "modelo/label_encoder.pkl"


def load_data():
    print("=== Carganfo dataset ===\n")

    X = []
    y = []

    for class_name in CLASSES:
        filepath = os.path.join(DATASET_DIR, f"{class_name}.npy")

        if not os.path.exists(filepath):
            print(f"Advertencia: {class_name}.npy no encontrado")
            continue

        print(f"Cargando {class_name}...", end=" ")
        data = np.load(filepath)

        # Normalizar píxeles a rango [0, 1]
        data = data.astype("float32") / 255.0

        X.append(data)
        y.extend([class_name] * len(data))

        print(f"{len(data)} muestras")

    X = np.vstack(X)
    y = np.array(y)

    print(f"\n Total: {len(X):,} imágenes de {len(CLASSES)} clases")
    print(f" Shape de cada imagen: {X.shape[1:]} píxeles\n")

    return X, y


def train_model(X, y):
    """Entrena el clasificador Random Forest"""
    print("=== Preparando Datos ===\n")

    # Codificar labels a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,  # Mantener proporción de clases
    )

    print(f"Entrenamiento: {len(X_train):,} muestras")
    print(f"Prueba: {len(X_test):,} muestras\n")

    # Entrenar Random Forest
    print("=== Entrenando Random Forest ===\n")
    print("Configuración:")
    print("  - n_estimators: 100")
    print("  - max_depth: 20")
    print("  - random_state: 42\n")

    clf = RandomForestClassifier(
        n_estimators=50,  # Número de árboles
        max_depth=20,  # Profundidad máxima
        random_state=42,
        n_jobs=-1,  # Usar todos los cores
        verbose=1,  # Mostrar progreso
    )

    clf.fit(X_train, y_train)

    print("\n✓ Entrenamiento completado\n")

    # Evaluar
    print("=== Evaluando Modelo ===\n")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")

    # Reporte detallado por clase
    print("Reporte por clase:")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, digits=3
        )
    )

    return clf, label_encoder


def save_model(clf, label_encoder):
    """Guarda el modelo y el encoder"""
    print("=== Guardando Modelo ===\n")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"✓ Modelo guardado en: {MODEL_PATH}")

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Encoder guardado en: {ENCODER_PATH}")

    # Mostrar tamaño
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n📦 Tamaño del modelo: {model_size:.1f} MB")


def main():
    print("\n" + "=" * 50)
    print("        EIKASIA - Entrenamiento de Modelo")
    print("=" * 50 + "\n")

    # 1. Cargar datos
    X, y = load_data()

    # 2. Entrenar
    clf, label_encoder = train_model(X, y)

    # 3. Guardar
    save_model(clf, label_encoder)

    print("\n" + "=" * 50)
    print("           ✓ Proceso Completado")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
