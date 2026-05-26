import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tp_2.addons.functions import perceptron_multicapa, evaluar_perceptron_multicapa
from utils.helpers import (
    cargar_txt,
    join_group_of_lists,
    outputs_multicapa,
    split_test_data,
    what_number_is,
)

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
    N = [0.1, 0.05, 0.01]
    B = [0.5, 0.75, 1.0]
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

    modelos_entrenados_pt2 = []
    for mode_idx in range(2):
        for n in N:
            for b in B:
                tanh_mode = mode_idx == 0
                tanh_mode_range = -1 if tanh_mode else 0
                layers = [35, 15, 2]
                COTA = 1000 # Aumentamos para PT2

                outputs_odd_even = outputs_multicapa(2, min=tanh_mode_range, max=1)
                impar = outputs_odd_even[0]
                par = outputs_odd_even[1]

                # Construimos el dataset etiquetado completo (cada fila tamaño 37)
                labeled_data = []
                for idx in range(len(number_bits)):
                    target = par if idx % 2 == 0 else impar
                    labeled_data.append(number_bits[idx] + target)

                # Dividimos los datos etiquetados
                test_data, train_data = split_test_data(labeled_data, shuffle_data=True, test_size=0.1)

                w, error, iterations = perceptron_multicapa(
                    train_data, n=n, COTA=COTA, β=b, layers=layers, tanh_mode=tanh_mode
                )
                modelos_entrenados_pt2.append(
                    {
                        "n": n,
                        "iterations": iterations,
                        "b": b,
                        "tanh_mode": tanh_mode,
                        "layers": layers,
                        "w": w,
                        "error_min": error,
                        "test_data": test_data,
                    }
                )
    
    mejor_resultado_pt2 = {"error_min": float("inf"), "model": {}, "result": []}
    for r in modelos_entrenados_pt2:
        if r["error_min"] <= mejor_resultado_pt2["error_min"]:
            mejor_resultado_pt2["error_min"] = r["error_min"]
            mejor_resultado_pt2["model"] = r

    hiperparams = mejor_resultado_pt2["model"]
    w = hiperparams["w"]
    b = hiperparams["b"]
    tanh_mode = hiperparams["tanh_mode"]
    test_data = hiperparams["test_data"]

    print(f"\n--- Evaluación PT2 (Par/Impar) ---")
    aciertos_pt2 = 0
    for test_pattern in test_data:
        num_outputs = 2
        test_x = test_pattern[:-num_outputs]
        test_y = test_pattern[-num_outputs:]
        number = what_number_is(test_x, number_bits)
        predicted = evaluar_perceptron_multicapa(
            test_pattern, w, num_outputs, β=b, tanh_mode=tanh_mode
        )
        test_result = {"number": number, "test_out": test_y, "predicted": predicted}
        mejor_resultado_pt2["result"].append(test_result)
        
        if np.argmax(test_y) == np.argmax(predicted):
            aciertos_pt2 += 1
            
    accuracy_pt2 = (aciertos_pt2 / len(test_data)) * 100 if len(test_data) > 0 else 0
    mejor_resultado_pt2["accuracy"] = accuracy_pt2
    
    modelos_entrenados_pt3 = []
    for mode_idx in range(2):
        for n in N:
            for b in B:
                tanh_mode = mode_idx == 0
                tanh_mode_range = -1 if tanh_mode else 0
                layers = [35, 15, 10]
                COTA = 30000 # PT3 es mucho más complejo

                base10_output = outputs_multicapa(10, min=tanh_mode_range, max=1)

                # Construimos el dataset etiquetado completo (cada fila tamaño 37)
                labeled_data = []
                for idx in range(len(number_bits)):
                    labeled_data.append(number_bits[idx] + base10_output[idx])

                # Dividimos los datos etiquetados
                test_data, train_data = split_test_data(labeled_data, shuffle_data=True, test_size=0.1)

                w, error, iterations = perceptron_multicapa(
                    train_data, n=n, COTA=COTA, β=b, layers=layers, tanh_mode=tanh_mode
                )
                modelos_entrenados_pt3.append(
                    {
                        "n": n,
                        "iterations": iterations,
                        "b": b,
                        "tanh_mode": tanh_mode,
                        "layers": layers,
                        "w": w,
                        "error_min": error,
                        "test_data": test_data,
                    }
                )

    mejor_resultado_tp3 = {"error_min": float("inf"), "model": {}, "result": []}
    for r in modelos_entrenados_pt3:
        if r["error_min"] <= mejor_resultado_tp3["error_min"]:
            mejor_resultado_tp3["error_min"] = r["error_min"]
            mejor_resultado_tp3["model"] = r

    hiperparams = mejor_resultado_tp3["model"]
    w = hiperparams["w"]
    b = hiperparams["b"]
    tanh_mode = hiperparams["tanh_mode"]
    test_data = hiperparams["test_data"]

    print(f"\n--- Evaluación PT3 (Identificación 0-9) ---")
    aciertos_pt3 = 0
    for test_pattern in test_data:
        num_outputs = 10
        test_x = test_pattern[:-num_outputs]
        test_y = test_pattern[-num_outputs:]
        number = what_number_is(test_x, number_bits)
        predicted = evaluar_perceptron_multicapa(
            test_pattern, w, num_outputs, β=b, tanh_mode=tanh_mode
        )
        test_result = {"number": number, "test_out": test_y, "predicted": predicted}
        mejor_resultado_tp3["result"].append(test_result)
        
        if np.argmax(test_y) == np.argmax(predicted):
            aciertos_pt3 += 1
        print(f"Dígito {number}: {'OK' if np.argmax(test_y) == np.argmax(predicted) else 'ERROR'} (Esperado: {np.argmax(test_y)}, Predicho: {np.argmax(predicted)})")

    accuracy_pt3 = (aciertos_pt3 / len(test_data)) * 100 if len(test_data) > 0 else 0
    mejor_resultado_tp3["accuracy"] = accuracy_pt3
    print(f"Accuracy PT3: {accuracy_pt3:.2f}%")
    
    # Limpieza de datos pesados para el print final
    for res in [mejor_resultado_pt2, mejor_resultado_tp3]:
        if "model" in res:
            res["model"].pop("w", None)
            res["model"].pop("test_data", None)

    return {
        #"pt1": res_error_min_tp1,
        "pt2": mejor_resultado_pt2,
        "pt3": mejor_resultado_tp3,
    }


# TODO: poner print para el conj de prueba
