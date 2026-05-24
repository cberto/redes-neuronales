import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tp_2.addons.functions import (
    perceptron_multicapa,
)
from utils.helpers import cargar_txt, split_test_data, join_group_of_lists

# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tp2():
    # Si usamos Tanh, los targets -1 y 1 están bien.
    # Si usamos Logística (tanh_mode=False), deberíamos cambiar los -1 por 0.
    # data_y = np.array([[-1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, 1, 1]])
    # data_xor = np.array([[-1, 1, 1], [1, -1, 1], [-1, -1, -1], [1, 1, -1]])

    # config = {
    #     "y": [
    #         {
    #             "cota": 10000,
    #             "tanh_mode": True,
    #             "layers": [2, 2, 1],
    #         },
    #     ],
    #     "xor": [
    #         {
    #             "cota": 10000,
    #             "tanh_mode": True,
    #             "layers": [2, 2, 1],
    #         },
    #     ],
    # }
    # N = [0.1, 0.05, 0.01]
    # B = [0.5, 0.75, 1.0]
    # # NO USAMOS LOGISTICA XQ LA SALIDA ES 1;-1
    # result_pt1 = {}
    # data_pt1 = {"y": data_y, "xor": data_xor}
    # for n in N:
    #     for b in B:
    #         for config_type, values in config.items():
    #             for config_data in values:
    #                 train_data = data_pt1[config_type]
    #                 COTA = config_data["cota"]
    #                 tanh_mode = config_data["tanh_mode"]
    #                 layers = config_data["layers"]
    #                 w, error, iter = perceptron_multicapa(
    #                     train_data, n, COTA, b, layers, tanh_mode
    #                 )
    #                 type = f"Taza de aprendizaje:{n}; Constante Beta:{b}"
    #                 print(f"{config_type} con n={n}, b={b} - error final: {error:.4f}")
    #                 if config_type not in result_pt1:
    #                     result_pt1[config_type] = {}
    #                 if type not in result_pt1[config_type]:
    #                     result_pt1[config_type][type] = []
    #                 result_pt1[config_type][type].append(
    #                     {
    #                         "n": n,
    #                         "cota": COTA,
    #                         "b": b,
    #                         "tanh_mode": tanh_mode,
    #                         "layers": layers,
    #                         "w": w,
    #                         "error_min": error,
    #                         "iterations": iter,
    #                     }
    #                 )
    # res_error_min_tp1 = {"error_min": float("inf"), "result": {}}
    # for res_tp1 in result_pt1.values():
    #     for res in res_tp1.values():
    #         for r in res:
    #             if r["error_min"] < res_error_min_tp1["error_min"]:
    #                 res_error_min_tp1["error_min"] = r["error_min"]
    #                 res_error_min_tp1["result"] = r

    path_sal = "./Documento/TP2-ej3-mapa-de-pixeles-digitos-decimales.txt"
    datos_sal = cargar_txt(path_sal)
    datos_sal = np.loadtxt(datos_sal, dtype=float)

    # Agrupamos las filas para obtener los 10 patrones de 35 bits
    number_bits = join_group_of_lists(datos_sal, 7)
    
    # Creamos el conjunto de entrenamiento con 35 entradas y 2 salidas (par/impar)
    training_data = []
    for i in range(len(number_bits)):
        number_bits[i].append(1.0 if i % 2 == 0 else -1.0)

    

    # Configuramos la red: 35 entradas, 15 neuronas ocultas (ejemplo) y 2 salidas
    w, error, _ = perceptron_multicapa(
        number_bits, n=0.1, COTA=10000, β=1.0, layers=[35, 15, 2], tanh_mode=True
    )

    print(f"Entrenamiento de dígitos completado. Error final: {error}")
    return w, error, number_bits


# TODO: poner print para el conj de prueba
