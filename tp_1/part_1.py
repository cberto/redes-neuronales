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

    y = { "data": data_y, "type": "y"}
    xor = { "data": data_xor, "type": "xor"}
    
    COTA = 2000
    n = 0.1
    b = 1
    tanh = False

    result_pt1 = []
    for data in [y, xor]:
        w, error_min, iterations = perceptron_simple(data["data"], n, COTA, b, tanh)
        # plot_perceptron(w, data["data"], error_min, iterations)
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
    
    data_pt2 = np.zeros((len(datos_ent), 4))
    
    for i in range(len(datos_ent)):
        registro = []
        registro.extend(datos_ent[i])
        registro.append(datos_sal[i])
        data_pt2[i] = registro

    # Convertimos los valores continuos (0-100) en clases (-1 y 1)
    result_pt2 = []
    umbral = 50
    for i in range(len(data_pt2)):
        if data_pt2[i][-1] >= umbral:  # Si la clase es 1 (>=50), la dejamos como 1
            data_pt2[i][-1] = 1
        else:  # Si la clase es -1 (<50), la dejamos como -1
            data_pt2[i][-1] = -1

    test, train = split_test_data(data_pt2)

    config = {
        "lineal": {"COTA": 8000, "n": 0.1, "b": 1, "tanh": False},
        "tanh": {"COTA": 8000, "n": 0.1, "b": 0.5, "tanh": True},
        "logis": {"COTA": 8000, "n": 0.1, "b": 0.5, "tanh": False},
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

            copy_test = test[i].copy()
            copy_test.append(tested)
            result_test[type].append(copy_test)

        deseada = 3
        obtenida = 4
        val_medio = 0.0

        matrix, _, str_matrix = confusion_matrix(
            result_test[type], deseada, obtenida, val_medio
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
