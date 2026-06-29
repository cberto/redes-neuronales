"""
Módulo con funciones de utilidad para cargar y procesar imágenes
de la competencia Dogs vs. Cats de Kaggle.

Incluye funciones para:
- Descargar el dataset desde Kaggle (usa kagglehub)
- Descomprimir los archivos zip
- Cargar y preprocesar imágenes
- Visualizar lotes de imágenes
- Construir modelo CNN con Transfer Learning (ResNet50)
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from random import shuffle
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parámetros por defecto (se pueden sobreescribir)
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 64
NO_EPOCHS = 20
NUM_CLASSES = 2
TEST_SIZE = 0.5
RANDOM_STATE = 2018
SAMPLE_SIZE = 100  # cantidad de imágenes de train a usar

# ---------------------------------------------------------------------------
# Descarga desde Kaggle
# ---------------------------------------------------------------------------
def download_kaggle_dataset(dest_dir: str = "./datasets") -> str:
    """
    Descarga la competencia dogs-vs-cats-redux-kernels-edition.
    
    Primero intenta con kagglehub. Si falla por autenticación, usa el CLI
    `kaggle competitions download`. Extrae los zips a `dest_dir`.
    Retorna la ruta local con los directorios train/ y test/.
    """
    import subprocess
    import shutil
    
    abs_dest = os.path.abspath(dest_dir)
    train_dir = os.path.join(abs_dest, "train")
    test_dir = os.path.join(abs_dest, "test")
    
    # Si ya están extraídos, salir rápido
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Usando datasets existentes en: {abs_dest}")
        return abs_dest
    
    competition = "dogs-vs-cats-redux-kernels-edition"
    zip_path = os.path.join(abs_dest, f"{competition}.zip")
    
    os.makedirs(abs_dest, exist_ok=True)
    
    # 1. Intentar con kagglehub
    try:
        import kagglehub
        print("Intentando descargar con kagglehub...")
        cache_path = kagglehub.competition_download(competition)
        print(f"Dataset descargado en caché: {cache_path}")
        
        # Extraer zips desde caché a dest_dir
        for name in ("train.zip", "test.zip"):
            src_zip = os.path.join(cache_path, name)
            if os.path.exists(src_zip):
                print(f"Extrayendo {name}...")
                with zipfile.ZipFile(src_zip, "r") as z:
                    z.extractall(abs_dest)
        
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            return abs_dest
    except Exception as e:
        print(f"kagglehub falló ({e}), usando CLI kaggle...")
    
    # 2. Fallback: CLI kaggle
    if not os.path.exists(zip_path):
        print(f"Descargando con: kaggle competitions download -c {competition}")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition],
            cwd=abs_dest,
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError(
                "No se pudo descargar el dataset. "
                "Asegurate de tener el CLI de Kaggle instalado y autenticado:\n"
                "  pip install kaggle\n"
                "  kaggle auth login"
            )
    
    # 3. Extraer zip(s)
    if os.path.exists(zip_path):
        print(f"Extrayendo {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(abs_dest)
    
    # 4. Extraer zips anidados (train.zip, test.zip dentro del zip principal)
    for name in ("train.zip", "test.zip"):
        nested_zip = os.path.join(abs_dest, name)
        if os.path.exists(nested_zip) and not os.path.exists(
            os.path.join(abs_dest, name.replace(".zip", ""))
        ):
            print(f"Extrayendo {name} anidado...")
            with zipfile.ZipFile(nested_zip, "r") as z:
                z.extractall(abs_dest)
    
    if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
        raise FileNotFoundError(
            f"No se encontraron las carpetas train/ y test/ en {abs_dest}"
        )
    
    return abs_dest

# ---------------------------------------------------------------------------
# Extracción de zip
# ---------------------------------------------------------------------------
def extract_data(zip_path: str, extract_to: str = "./data"):
    """
    Extrae un archivo zip en el directorio `extract_to`.
    """
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"Extraído {zip_path} → {extract_to}")

def prepare_data(source_dir: str, dest_dir: str = "./data") -> dict:
    """
    Busca archivos .zip dentro de source_dir (train.zip, test.zip)
    y los extrae en dest_dir/train y dest_dir/test.
    Retorna un dict con las rutas de las carpetas de imágenes.
    """
    train_zip = os.path.join(source_dir, "train.zip")
    test_zip  = os.path.join(source_dir, "test.zip")

    train_dir = os.path.join(dest_dir, "train")
    test_dir  = os.path.join(dest_dir, "test")

    if not os.path.exists(train_dir):
        print("Extrayendo train.zip...")
        extract_data(train_zip, dest_dir)
    else:
        print(f"Usando carpeta existente: {train_dir}")

    if not os.path.exists(test_dir):
        print("Extrayendo test.zip...")
        extract_data(test_zip, dest_dir)
    else:
        print(f"Usando carpeta existente: {test_dir}")

    return {"train": train_dir, "test": test_dir}

# ---------------------------------------------------------------------------
# Etiquetado
# ---------------------------------------------------------------------------
def label_pet_image_one_hot_encoder(img: str):
    """
    Dado el nombre de archivo (ej: 'cat.0.jpg' o 'dog.1.jpg'),
    retorna [1,0] para cat y [0,1] para dog.
    """
    pet = img.split('.')[-3]
    if pet == 'cat':
        return [1, 0]
    elif pet == 'dog':
        return [0, 1]

# ---------------------------------------------------------------------------
# Carga y preprocesamiento
# ---------------------------------------------------------------------------
def process_data(data_image_list, data_folder: str, is_train: bool = True):
    """
    Procesa una lista de nombres de imágenes:
    - Lee cada imagen con OpenCV
    - Redimensiona a (IMG_SIZE, IMG_SIZE)
    - Si is_train=True, extrae la etiqueta del nombre del archivo
    - Si is_train=False, usa el id numérico como etiqueta

    Retorna una lista de [imagen_array, label_array].
    """
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(data_folder, img)
        if is_train:
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img_arr = cv2.imread(path, cv2.IMREAD_COLOR)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data_df.append([np.array(img_arr), np.array(label)])
    shuffle(data_df)
    return data_df

# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def plot_image_list_count(data_image_list):
    """Muestra un countplot de gatos vs perros."""
    labels = [img.split('.')[-3] for img in data_image_list]
    sns.countplot(x=labels)
    plt.title('Cats and Dogs')
    plt.show()

def show_images(data, is_test: bool = False):
    """Muestra una grilla 5×5 con las primeras 25 imágenes."""
    f, ax = plt.subplots(5, 5, figsize=(15, 15))
    for i, sample in enumerate(data[:25]):
        img_data = sample[0]
        label = np.argmax(sample[1])
        str_label = 'Dog' if label == 1 else 'Cat'
        if is_test:
            str_label = "None"
        ax[i // 5, i % 5].imshow(img_data)
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title(f"Label: {str_label}")
    plt.show()

# ---------------------------------------------------------------------------
# Gráficos de entrenamiento
# ---------------------------------------------------------------------------
def plot_accuracy_and_loss(train_model):
    """Grafica accuracy y loss de entrenamiento/validación."""
    hist = train_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))

    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()

    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()
