import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import cargar_txt, read_json, hex_a_binario
from tp_3.addons.functions import instancia_keras
# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tp3():
    
    path_sal = "./Documento/tp3.json"
    datos_sal = read_json(path_sal)
    matrix_binary = hex_a_binario(datos_sal)
    print(matrix_binary)
    #instancia_keras()
    return {}


# TODO: poner print para el conj de prueba
