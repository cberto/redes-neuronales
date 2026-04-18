# from utils.helpers import (
# )
# from tp_1.addons.functions import (
# )
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tp_1.addons.functions import evaluar_perceptron_simple, perceptron_simple
from utils.helpers import cargar_tp1_ej2, plot_perceptron, split_test_data

# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
_DOC = Path(__file__).resolve().parents[1] / "Documento"


def _inv_y_a_original(o_n, tipo, meta):
    """Deshace el escalado de y usado en entrenamiento (misma escala que el TXT)."""
    t = tipo.lower()
    if t == "lineal":
        return float(o_n) * meta["y_std"] + meta["y_mean"]
    if t == "tanh":
        return (float(o_n) + 1.0) * 0.5 * meta["span"] + meta["y_min"]
    if t in ("logis", "logistica"):
        return (float(o_n) - 0.1) / 0.8 * meta["span"] + meta["y_min"]
    raise ValueError(t)


def tp1():
    data_y = np.array([[-1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, 1, 1]])
    data_xor = np.array([[-1, 1, 1], [1, -1, 1], [-1, -1, -1], [1, 1, -1]])

    COTA = 2000
    n = 0.1
    b = 1
    tanh = False

    result_pt1 = []
    for data in [data_y, data_xor]:
        w, error_min, iterations = perceptron_simple(data, n, COTA, b, tanh)
        # plot_perceptron(w, data, error_min, iterations)
        result_pt1.append(
            {
                "w0": w[0],
                "w1": w[1],
                "w2": w[2],
                "w": w,
                "error_min": error_min,
                "iterations": iterations,
            }
        )

    datos_sal = _DOC / "TP1-ej2-Salida-deseada.txt"
    datos_ent = _DOC / "TP1-ej2-Conjunto-entrenamiento.txt"

    result_pt2 = []
    if datos_ent.exists() and datos_sal.exists():
        data_pt2 = cargar_tp1_ej2(datos_ent, datos_sal)

        test, train = split_test_data(data_pt2)
        train_arr = np.asarray(train, dtype=float)
        test_arr = np.asarray(test, dtype=float)

        X_tr, y_tr = train_arr[:, :3], train_arr[:, 3]
        X_te, y_te = test_arr[:, :3], test_arr[:, 3]

        x_mean = X_tr.mean(axis=0)
        x_std = X_tr.std(axis=0) + 1e-12
        X_tr_n = (X_tr - x_mean) / x_std
        X_te_n = (X_te - x_mean) / x_std

        y_mean = float(y_tr.mean())
        y_std = float(y_tr.std()) + 1e-12
        y_min, y_max = float(y_tr.min()), float(y_tr.max())
        span = y_max - y_min + 1e-12

        y_lineal = (y_tr - y_mean) / y_std
        y_tanh = 2.0 * (y_tr - y_min) / span - 1.0
        y_log = 0.1 + 0.8 * (y_tr - y_min) / span

        train_mats = {
            "lineal": np.column_stack([X_tr_n, y_lineal]),
            "tanh": np.column_stack([X_tr_n, y_tanh]),
            "logis": np.column_stack([X_tr_n, y_log]),
        }
        metas = {
            "lineal": {"y_mean": y_mean, "y_std": y_std},
            "tanh": {"y_min": y_min, "span": span},
            "logis": {"y_min": y_min, "span": span},
        }

        # Misma idea: dict por tipo; ej2_tipo solo engancha salida/derivada correctas en functions.py
        config = {
            "lineal": {"COTA": 2000, "n": 0.1, "b": 1, "tanh": False},
            "tanh": {"COTA": 2000, "n": 0.1, "b": 0.5, "tanh": True},
            "logis": {"COTA": 2000, "n": 0.1, "b": 0.5, "tanh": False},
            "activado": "sin activacion",
        }

        for tipo in ("lineal", "tanh", "logis"):
            w, error, it = perceptron_simple(
                train_mats[tipo],
                config[tipo]["n"],
                config[tipo]["COTA"],
                config[tipo]["b"],
                config[tipo]["tanh"],
                config["activado"],
                ej2_tipo=tipo,
            )
            preds = []
            for i in range(len(X_te_n)):
                o_n = evaluar_perceptron_simple(
                    X_te_n[i],
                    w,
                    config["activado"],
                    ej2_tipo=tipo,
                    b=config[tipo]["b"],
                )
                preds.append(_inv_y_a_original(o_n, tipo, metas[tipo]))
            preds = np.asarray(preds, dtype=float)
            mse_test = float(np.mean((y_te - preds) ** 2))

            result_pt2.append(
                {
                    "type": tipo,
                    "w": w.tolist() if hasattr(w, "tolist") else w,
                    "error_train": error,
                    "iter": it,
                    "mse_test": mse_test,
                }
            )
    else:
        print(f"Error: No se encontraron los archivos en {_DOC}")
        print(f"Verifica si la carpeta se llama 'Documento' o 'documento' (minúsculas)")

    return {
        "pt1": result_pt1,
        "pt2": result_pt2,
    }


# TODO: poner print para el conj de prueba
