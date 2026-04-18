# from utils.helpers import (
# )
# from tp_1.addons.functions import (
# )
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tp_1.addons.functions import (
    evaluar_perceptron_simple,
    perceptron_simple,
)
from utils.helpers import (
    accuracy_score,
    cargar_tp1_ej2,
    plot_perceptron,
    split_test_data,
    confusion_matrix,
)

# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
_DOC = Path(__file__).resolve().parents[1] / "Documento"


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

    # Carga de archivos después del for inicial
    datos_sal = _DOC / "TP1-ej2-Salida-deseada.txt"
    datos_ent = _DOC / "TP1-ej2-Conjunto-entrenamiento.txt"

    result_pt2 = []
    if datos_ent.exists() and datos_sal.exists():
        data_pt2 = cargar_tp1_ej2(datos_ent, datos_sal)

        # Binarización: Convertimos la salida deseada a 1 (activado) o -1 (no activado)
        # Usamos 50 como umbral (valor medio aproximado del archivo TXT)
        data_pt2[:, -1] = np.where(data_pt2[:, -1] > 50, 1.0, -1.0)

        test, train = split_test_data(data_pt2)

        config = {
            "lineal": {"COTA": 2000, "n": 0.1, "b": 1, "tanh": False},
            "tanh": {"COTA": 2000, "n": 0.1, "b": 0.5, "tanh": True},
            "logis": {"COTA": 2000, "n": 0.1, "b": 0.5, "tanh": False},
        }
        result_test = {"lineal": [], "tanh": [], "logis": []}
        for type in result_test.keys():
            w, error, iter = perceptron_simple(
                train,
                config[type]["n"],
                config[type]["COTA"],
                config[type]["b"],
                config[type]["tanh"],
            )
            for i in range(len(test)):
                tested = evaluar_perceptron_simple(test[i][:-1], w)

                copy_test = np.append(test[i], tested)
                result_test[type].append(copy_test)

            deseada = 3
            obtenida = 4
            activacion = 1
            
            matrix, _, str_matrix = confusion_matrix(
                result_test[type], deseada, obtenida, activacion
            )
            tp = matrix["tp"]
            tn = matrix["tn"]
            fp = matrix["fp"]
            fn = matrix["fn"]
            accuracy = accuracy_score(tp, tn, fp, fn)
            result_pt2.append(
                {
                    "type": type,
                    "w": w,
                    "error": error,
                    "iter": iter,
                    "conf_matrix": str_matrix,
                    "accuracy": accuracy,
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
