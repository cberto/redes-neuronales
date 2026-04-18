"""Carga de datos del ejercicio 2 del TP1 (archivos de texto en Documento/)."""

from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONJUNTO = _REPO_ROOT / "Documento" / "TP1-ej2-Conjunto-entrenamiento.txt"
_DEFAULT_SALIDA = _REPO_ROOT / "Documento" / "TP1-ej2-Salida-deseada.txt"


def cargar_tp1_ej2(
    conjunto_path=None,
    salida_path=None,
) -> np.ndarray:
    """
    Devuelve un array (p, 4): tres columnas de entrada y una de salida deseada (continua).
    """
    cpath = Path(conjunto_path) if conjunto_path else _DEFAULT_CONJUNTO
    spath = Path(salida_path) if salida_path else _DEFAULT_SALIDA
    x = np.loadtxt(cpath)
    y = np.loadtxt(spath)
    if y.ndim == 0:
        y = np.array([float(y)])
    elif y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Cantidad de filas distinta: X {x.shape[0]} vs y {y.shape[0]}"
        )
    return np.hstack([x, y])
