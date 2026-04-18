import numpy as np

from tp_1.addons.data_tp1 import cargar_tp1_ej2
from tp_1.addons.functions import (
    desnormalizar_minmax,
    evaluar_mse,
    excitacion,
    normalizar_salida_minmax,
    perceptron_lineal,
    perceptron_no_lineal,
    salida_no_lineal,
)
from utils.helpers import split_test_data


def _split_tp1_ej2(data, test_size=0.2):
    """Partición train/test con el mismo criterio que `split_test_data` del proyecto."""
    rows = data.tolist()
    test_rows, train_rows = split_test_data(rows, test_size=test_size)
    train = np.asarray(train_rows, dtype=float)
    test = np.asarray(test_rows, dtype=float)
    return train, test


def _mse_no_lineal_original(test_data, w, beta, y_min, y_max, tanh=True):
    """MSE en escala original: salida de la neurona en [-1,1] desnormalizada vs y del archivo."""
    x_raw = test_data[:, :-1]
    y_true = test_data[:, -1].astype(float)
    p = len(y_true)
    x = np.ones((p, x_raw.shape[1] + 1))
    x[:, 1:] = x_raw
    total = 0.0
    for i in range(p):
        h = excitacion(x[i], w)
        O = float(salida_no_lineal(h, beta, tanh))
        y_hat = float(desnormalizar_minmax(O, y_min, y_max))
        total += (y_true[i] - y_hat) ** 2
    return total / p


def tp1_ej2():
    data = cargar_tp1_ej2()
    train, test = _split_tp1_ej2(data, test_size=0.2)

    COTA = 50_000
    n = 0.001
    beta = 1.0
    tol = 0.05

    w_lin, err_lin, it_lin = perceptron_lineal(train, n=n, COTA=COTA, tol=tol)
    mse_train_lin = evaluar_mse(train, w_lin, lineal=True)
    mse_test_lin = evaluar_mse(test, w_lin, lineal=True)

    y_min = float(np.min(train[:, -1]))
    y_max = float(np.max(train[:, -1]))
    train_n, _, _ = normalizar_salida_minmax(train, y_min=y_min, y_max=y_max)
    w_nl, err_nl, it_nl = perceptron_no_lineal(
        train_n, n=n, COTA=COTA, beta=beta, tanh=True, tol=tol
    )
    mse_train_nl_norm = evaluar_mse(train_n, w_nl, beta=beta, lineal=False, tanh=True)
    mse_test_nl_orig = _mse_no_lineal_original(test, w_nl, beta, y_min, y_max, tanh=True)

    return {
        "lineal": {
            "mse_train": float(mse_train_lin),
            "mse_test": float(mse_test_lin),
            "iteraciones": int(it_lin),
            "mse_min_entrenamiento": float(err_lin),
        },
        "no_lineal": {
            "mse_train_normalizado": float(mse_train_nl_norm),
            "mse_test_escala_original": float(mse_test_nl_orig),
            "iteraciones": int(it_nl),
            "mse_min_entrenamiento_normalizado": float(err_nl),
            "y_min_train": y_min,
            "y_max_train": y_max,
            "beta": beta,
        },
        "split": {"train": len(train), "test": len(test)},
    }
