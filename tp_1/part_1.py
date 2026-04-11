# from utils.helpers import (
# )
# from tp_1.addons.functions import (
# )
import numpy as np
from tp_1.addons.functions import (
    perceptron_simple,
)


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
    ent = [3, 4]
    COTA = 2000
    n = 0.1
    print(perceptron_simple(data, n, COTA))
    return {}


# TODO: poner print para el conj de prueba
