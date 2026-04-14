# from utils.helpers import (
# )
# from tp_1.addons.functions import (
# )
import numpy as np
import matplotlib.pyplot as plt
from tp_1.addons.functions import (
    perceptron_simple,
)
from utils.helpers import plot_perceptron


def tp1():
    data = np.array(
        [
            [13, 9, 1],
            [5, 4, -1],
            [2, 8, 1],
            [6, 2, -1],
            [7, 3, -1],
            [8, 5, 1],
            [4, 6, 1],
            [9, 7, 1],
            [3, 1, -1],
            [10, 10, 1],
        ]
    )
    COTA = 2000
    n = 0.1
    b = 1
    tanh = False

    # Ejecución del perceptrón
    w, error_min, iterations = perceptron_simple(data, n, COTA, b, tanh)

    plot_perceptron(w, data, error_min, iterations)

    return {
        "w0": w[0],
        "w1": w[1],
        "w2": w[2],
        "w": w,
        "error_min": error_min,
        "iterations": iterations,
    }


# TODO: poner print para el conj de prueba
