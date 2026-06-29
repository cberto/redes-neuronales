"""
Descarga imágenes desde Kaggle y las organiza por categoría.

Uso desde main.py:
    from utils.preparar_datos import descargar

    descargar(
        api_command="competitions download -c dogs-vs-cats-redux-kernels-edition",
        dest_dir="./datasets",
        categorias=["cat", "dog"],
    )
"""

import os
import zipfile
import subprocess
import json
import shutil
from datetime import datetime


def descargar(api_command, dest_dir="./datasets", categorias=None, limite=100):
    """
    Descarga un dataset desde Kaggle y organiza las imágenes
    en subcarpetas según las categorías indicadas.

    Parámetros
    ----------
    api_command : str
        Comando de la API de Kaggle, ej:
        "competitions download -c dogs-vs-cats-redux-kernels-edition"
    dest_dir : str
        Directorio donde descargar (default: ./datasets)
    categorias : list[str] | None
        Lista de nombres de categorías (ej: ["cat", "dog"]).
        Si se pasa, busca imágenes cuyo nombre empiece con "categoria."
        y las copia a subcarpetas. Si es None, solo descarga y extrae.
    limite : int
        Cantidad máxima de imágenes por categoría (default: 100).
    """
    abs_dest = os.path.abspath(dest_dir)
    os.makedirs(abs_dest, exist_ok=True)

    # 1. Ejecutar el comando de Kaggle
    cmd = api_command.strip()
    if not cmd.startswith("kaggle "):
        cmd = f"kaggle {cmd}"
    if "-p " not in cmd:
        cmd = f"{cmd} -p {abs_dest}"

    print(f"Ejecutando: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(
            "¿Tenés instalado y autenticado el CLI de Kaggle?\n"
            "  pip install kaggle\n"
            "  kaggle auth login"
        )

    # 2. Extraer zips que estén en dest_dir
    for f in os.listdir(abs_dest):
        if f.endswith(".zip"):
            src_zip = os.path.join(abs_dest, f)
            print(f"Extrayendo {f}...")
            with zipfile.ZipFile(src_zip, "r") as z:
                z.extractall(abs_dest)

    # 3. Extraer zips anidados (train.zip, test.zip, etc.)
    for f in os.listdir(abs_dest):
        if f.endswith(".zip"):
            nested_zip = os.path.join(abs_dest, f)
            extract_to = os.path.join(abs_dest, f.replace(".zip", ""))
            if not os.path.exists(extract_to):
                print(f"Extrayendo {f} anidado...")
                with zipfile.ZipFile(nested_zip, "r") as z:
                    z.extractall(abs_dest)

    # 4. Si hay categorías, organizar por subcarpetas
    if categorias:
        carpetas_img = []
        for root, dirs, files in os.walk(abs_dest):
            if any(f.endswith((".jpg", ".png", ".jpeg")) for f in files):
                carpetas_img.append(root)

        for carpeta in carpetas_img:
            archivos = os.listdir(carpeta)
            tiene_alguna = any(
                any(f.startswith(f"{cat}.") for f in archivos)
                for cat in categorias
            )
            if not tiene_alguna:
                continue

            print(f"Organizando por categorías {categorias} (máx. {limite} c/u)...")
            contadores = {cat: 0 for cat in categorias}
            for cat in categorias:
                subdir = os.path.join(abs_dest, cat)
                os.makedirs(subdir, exist_ok=True)

            for f_name in archivos:
                src = os.path.join(carpeta, f_name)
                if not os.path.isfile(src):
                    continue
                for cat in categorias:
                    if not f_name.startswith(f"{cat}."):
                        continue
                    if contadores[cat] >= limite:
                        break
                    shutil.copy2(src, os.path.join(abs_dest, cat, f_name))
                    contadores[cat] += 1
                    break

    # 5. Resumen
    print("\n✅ Descarga finalizada.")
    for item in sorted(os.listdir(abs_dest)):
        ruta = os.path.join(abs_dest, item)
        if os.path.isdir(ruta):
            cant = len(os.listdir(ruta))
            print(f"  📁 {item}/  ({cant} archivos)")
        elif item.endswith(".zip"):
            size = os.path.getsize(ruta) // (1024 * 1024)
            print(f"   {item}  ({size} MB)")