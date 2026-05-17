import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tp_2.addons.functions import (
    evaluar_perceptron_simple,
    perceptron_multicapa,
)
from utils.helpers import (
    accuracy_score,
    cargar_txt,
    split_test_data,
    confusion_matrix,
)

# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tp2():
    data_y = np.array([[-1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, 1, 1]])
    data_xor = np.array([[-1, 1, 1], [1, -1, 1], [-1, -1, -1], [1, 1, -1]])

    y = {"data": data_y, "type": "y"}
    xor = {"data": data_xor, "type": "xor"}
    
    # Carga de archivos después del for inicial
    # path = "./datos.txt"
    # datos = cargar_txt(path)

    #datos = np.loadtxt(datos, dtype=float)
    perceptron_multicapa(data_y, 0.1, 20, 0.1, [2, 2, 2])
    

    return {
    }


# TODO: poner print para el conj de prueba
