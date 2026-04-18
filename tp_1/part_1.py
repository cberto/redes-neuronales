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
from utils.helpers import cargar_tp1_ej2, plot_perceptron, split_test_data, confusion_matrix
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
        plot_perceptron(w, data, error_min, iterations)
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
        
        test, train = split_test_data(data_pt2)  
        COTA = 2000
        n = 0.1
        b = 1
        tanh = False

        w_li, error_lin, iter_lin = perceptron_simple(train, n, COTA, b, tanh)
        result_test = {"lineal": [], "tanh": [], "logis": []}
        for i in range(len(test)):
            tested = evaluar_perceptron_simple(test[i][:-1], w_li)

            copy_test = np.append(test[i], tested)
            result_test["lineal"].append(copy_test)
        
        confusion_matrix(result_test["lineal"], 3, 4, 1)
            
        b = 0.5
        w_nli_tanh, error_nlin_tanh, iter_nlin_tanh = perceptron_simple(train, n, COTA, b, tanh)
        
        b = 0.5
        tanh = True
        w_nli_logis, error_nlin_logis, iter_nlin_logis = perceptron_simple(train, n, COTA, b, tanh)
        
        
        result_pt1.append({
            "w": w_li,
            "error_min": error_lin,
            "iterations": iter_lin,
            "tipo": "TP1-Ej2"
        })
    else:
        print(f"Error: No se encontraron los archivos en {_DOC}")
        print(f"Verifica si la carpeta se llama 'Documento' o 'documento' (minúsculas)")

    return result_pt1



# TODO: poner print para el conj de prueba
