import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tp_1.addons.functions import (
    evaluar_perceptron_simple,
    perceptron_simple,
)
from utils.helpers import (
    accuracy_score,
    cargar_txt,
    plot_perceptron,
    split_test_data,
    confusion_matrix,
)

# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tp1():
    data_y = np.array([[-1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, 1, 1]])
    data_xor = np.array([[-1, 1, 1], [1, -1, 1], [-1, -1, -1], [1, 1, -1]])

    y = {"data": data_y, "type": "y"}
    xor = {"data": data_xor, "type": "xor"}

    COTA = 2000
    n = 0.1
    b = None
    tanh = False

    result_pt1 = []
    for data in [y, xor]:
        w, error_min, iterations = perceptron_simple(data["data"], n, COTA, b, tanh)
        plot_perceptron(w, data["data"], error_min, iterations)
        result_pt1.append(
            {
                "type": data["type"],
                "w0": w[0],
                "w1": w[1],
                "w2": w[2],
                "w": w,
                "error_min": error_min,
                "iterations": iterations,
            }
        )

    # Carga de archivos después del for inicial
    path_sal = "./Documento/TP1-ej2-Salida-deseada.txt"
    path_ent = "./Documento/TP1-ej2-Conjunto-entrenamiento.txt"
    datos_sal = cargar_txt(path_sal)
    datos_ent = cargar_txt(path_ent)

    datos_ent = np.loadtxt(datos_ent, dtype=float)
    datos_sal = np.loadtxt(datos_sal, dtype=float)

    # Determinamos un umbral para separar clases en los datos originales
    # Si el valor de salida es mayor al promedio, lo consideramos 'OTORGADO' (1.0)

    result_pt2 = []
    config = {
        "lineal": {
            "COTA": 1000,
            "n": 0.1,
            "activation": "lineal",
            "b": None,
            "tanh_mode": False,
        },
        "tanh_mode": {
            "COTA": 1000,
            "n": 0.1,
            "activation": None,
            "b": 0.5,
            "tanh_mode": True,
        },
        "logis": {
            "COTA": 1000,
            "n": 0.1,
            "activation": None,
            "b": 0.5,
            "tanh_mode": False,
        },
    }
    # reescalar valores a t para 0-1
    # logi -1 - 1

    result_test = {"lineal": [], "tanh_mode": [], "logis": []}
    for type in result_test.keys():
        is_logis = type == "logis"

        data_pt2 = []
        min = np.min(datos_sal)
        max = np.max(datos_sal)
        range_min = 0 if is_logis else -1
        range_max = 1
        for i in range(len(datos_ent)):
            # Si supera el umbral es 1, sino es el mínimo del rango

            scaled = (datos_sal[i] - min) / (max - min) * (
                range_max - range_min
            ) + range_min

            fila = []
            fila.extend(datos_ent[i])  # Agregamos las características
            fila.append(scaled)
            data_pt2.append(fila)

        test, train = split_test_data(data_pt2)

        n = config[type]["n"]
        COTA = config[type]["COTA"]
        b = config[type]["b"]
        tanh_mode = config[type]["tanh_mode"]
        activacion = config[type]["activation"]
        w, error, iter = perceptron_simple(train, n, COTA, b, tanh_mode, activacion)

        for i in range(len(test)):
            tested = evaluar_perceptron_simple(
                test[i][:-1], w, b, tanh_mode, activacion
            )

            copy_test = test[i].copy()
            copy_test.append(tested)
            result_test[type].append(copy_test)

        # posicion dentro del array
        # 0,  1,   2,     3,              4
        # x1, x2, x3, clase, clase_predicha
        deseada = 3
        obtenida = 4

        condicion = 1.0
        val_medio = 0.5 if is_logis else 0.0

        matrix, _, str_matrix = confusion_matrix(
            result_test[type], deseada, obtenida, condicion, val_medio
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

    return {
        "pt1": result_pt1,
        "pt2": result_pt2,
    }


# TODO: poner print para el conj de prueba
