# i = 0
# w = zeros(N+1, 1)
# error = 1
# error min = p * 2
# while error > 0 ∧ i < COTA
# Tomar un n´umero i x al azar entre 1 y p
# Calcular la exitaci´on h = x[i x].w
# Calcular la activaci´on O = signo(h)
# ∆w = η ∗ (y[i x] − O).x[i x]
# w = w + ∆w
# error = CalcularError(x, y, w, p)
# if error < error min
# error min = error
# w min = w
# end
# i = i + 1
# end
import numpy as np
import math as Math


# COTA es hiperparametro de cantidad de iteraciones, p es cantidad de patrones, n es tasa de aprendizaje
def perceptron_simple(data, n, COTA, b=None, tanh=False):
    iterations = 0
    p = len(data)
    y = data[:, -1]

    # Agregamos manualmente el "1" del bias a cada entrada x
    # Si x era [13, 9], ahora será [1, 13, 9]
    x_raw = data[:, :-1]
    x = np.ones((p, x_raw.shape[1] + 1))

    x[:, 1:] = x_raw
    # Ahora w tiene un tamaño extra para incluir a w0
    num_features = x.shape[1]
    w = np.zeros(num_features)

    error = 1
    error_min = float("inf")
    w_min = w.copy()

    while error > 0 and iterations < COTA:
        pos = np.random.randint(0, p)
        x_μ = x[pos]
        y_μ = y[pos]
        h = excitacion(x_μ, w)
        O = activacion(h)

        # Update the entire weight vector, not just one index
        dw = delta_w(n, y_μ, O, x_μ, h, b=b, tanh=tanh)

        # Sumamos el vector delta al vector de pesos elemento por elemento
        for w_i in range(len(w)):
            w[w_i] = w[w_i] + dw[w_i]

        error = calcular_error(x, y, w, p)
        if error < error_min:
            error_min = error
            w_min = w.copy()
        iterations += 1
    return w_min, error_min, iterations


def calcular_error(x, y, w, p):
    total_error = 0
    for pos in range(p):
        x_μ = x[pos]
        y_μ = y[pos]
        h = excitacion(x_μ, w)  # calcular excitacion para el patron i
        O = activacion(h)
        total_error += (y_μ - O) ** 2
    return 0.5 * total_error

#β(1 − g2(h))
def sigmoide_tanh(b, h):
    output = Math.tanh(b * h)
    return b * (1 - output**2)

# g′(h) = 2βg(h)(1 − g(h)) 
def sigmoidea_logica(b, h):
    g_h = 1/(1+ Math.exp(-2*b*h))
    return 2 * b * g_h * (1 - g_h)


     



def delta_w(n, y_μ, O, x_μ, h, b=None, tanh=False):
    """
    Calcula el ajuste de pesos de forma explícita.
    No es un producto de matrices, es un ESCALAR multiplicado por un VECTOR.
    """
    # 1. Calculamos el factor común (escalar): tasa de aprendizaje * error
    g_h = 1
    if b is not None:
        # tanh=True -> derivada de tanh(βh); False -> derivada de sigmoidea logística g(h)
        g_h = sigmoide_tanh(b, h) if tanh else sigmoidea_logica(b, h)

    factor_aprendizaje = n * (y_μ - O) * g_h

    # 2. Creamos un vector vacío para el resultado (mismo tamaño que las entradas)
    dw = np.zeros(len(x_μ))

    # 3. Multiplicamos el factor por cada componente de la entrada x_μ
    # Esto nos da el ajuste individual para cada peso w_j
    for i in range(len(x_μ)):
        dw[i] = factor_aprendizaje * x_μ[i]

    return dw




def excitacion(x_μ, w):
    """Excitación h = x · w para un solo patrón x (vector fila) y pesos w."""
    h = 0.0
    for i in range(len(x_μ)):
        x_i = float(x_μ[i])
        w_i = float(w[i])
        h += x_i * w_i
    return h


def activacion(h):
    """Activación O = signo(h) para el perceptrón (±1, 0 si h == 0)."""
    # Si h es positivo, la neurona se activa (1)
    if h > 0:
        return 1.0
    # Si h es negativo, la neurona se inhibe (-1)
    elif h < 0:
        return -1.0
    # Si es exactamente 0, se queda neutral
    return 0.0


def _construir_x_con_bias(data):
    """data: (p, n_features+1) con última columna y. Devuelve x (p, n+1) con columna 1 para bias."""
    p = len(data)
    y = data[:, -1]
    x_raw = data[:, :-1]
    x = np.ones((p, x_raw.shape[1] + 1))
    x[:, 1:] = x_raw
    return x, y


def calcular_error_mse_lineal(x, y, w, p):
    """Error cuadrático medio (sin el 1/2) para salida lineal O = h."""
    total = 0.0
    for pos in range(p):
        h = excitacion(x[pos], w)
        total += (y[pos] - h) ** 2
    return total / p


def perceptron_lineal(data, n, COTA, tol=1e-4):
    """
    Perceptrón simple lineal: salida O = h = w·x (regresión).
    Actualización estocástica: Δw = η (y − h) x.
    """
    p = len(data)
    x, y = _construir_x_con_bias(data)
    w = np.zeros(x.shape[1])

    error_min = float("inf")
    w_min = w.copy()
    iterations = 0
    error = float("inf")

    while error > tol and iterations < COTA:
        pos = np.random.randint(0, p)
        x_μ = x[pos]
        y_μ = float(y[pos])
        h = excitacion(x_μ, w)
        factor = n * (y_μ - h)
        w = w + factor * x_μ

        error = calcular_error_mse_lineal(x, y, w, p)
        if error < error_min:
            error_min = error
            w_min = w.copy()
        iterations += 1

    return w_min, error_min, iterations


def _activacion_tanh_out(beta, h):
    return Math.tanh(beta * h)


def salida_no_lineal(h, beta, tanh=True):
    """Salida O de la neurona: tanh(βh) o sigmoide logística."""
    if tanh:
        return _activacion_tanh_out(beta, h)
    return 1.0 / (1.0 + Math.exp(-2 * beta * h))


def normalizar_salida_minmax(data, y_min=None, y_max=None):
    """
    Escala la última columna (y) a [-1, 1]. Si y_min/y_max vienen dados (p. ej. del train),
    aplica la misma transformación sin filtrar información del test.
    """
    d = np.array(data, dtype=float, copy=True)
    y = d[:, -1]
    if y_min is None:
        y_min = float(np.min(y))
    if y_max is None:
        y_max = float(np.max(y))
    span = y_max - y_min if y_max > y_min else 1.0
    d[:, -1] = 2.0 * (y - y_min) / span - 1.0
    return d, y_min, y_max


def desnormalizar_minmax(y_norm, y_min, y_max):
    span = y_max - y_min if y_max > y_min else 1.0
    return (np.asarray(y_norm, dtype=float) + 1.0) * 0.5 * span + y_min


def perceptron_no_lineal(data, n, COTA, beta, tanh=True, tol=1e-4):
    """
    Perceptrón simple no lineal: O = tanh(βh) o sigmoide logística g(h).
    Gradiente estocástico para E = ½(y − O)²: Δw = η (y − O) g'(h) x.
    Se asume que `data` ya tiene la salida en el rango apropiado (p. ej. [-1,1]).
    """
    p = len(data)
    x, y = _construir_x_con_bias(data)
    w = np.zeros(x.shape[1])

    error_min = float("inf")
    w_min = w.copy()
    iterations = 0
    error = float("inf")

    while error > tol and iterations < COTA:
        pos = np.random.randint(0, p)
        x_μ = x[pos]
        y_μ = float(y[pos])
        h = excitacion(x_μ, w)
        if tanh:
            O = salida_no_lineal(h, beta, tanh=True)
            g_prime = sigmoide_tanh(beta, h)
        else:
            O = salida_no_lineal(h, beta, tanh=False)
            g_prime = sigmoidea_logica(beta, h)

        factor = n * (y_μ - O) * g_prime
        w = w + factor * x_μ

        error = _error_mse_no_lineal(x, y, w, p, beta, tanh)
        if error < error_min:
            error_min = error
            w_min = w.copy()
        iterations += 1

    return w_min, error_min, iterations


def _error_mse_no_lineal(x, y, w, p, beta, tanh):
    total = 0.0
    for pos in range(p):
        h = excitacion(x[pos], w)
        O = salida_no_lineal(h, beta, tanh)
        total += (float(y[pos]) - O) ** 2
    return total / p


def evaluar_mse(data, w, beta=None, lineal=True, tanh=True):
    """MSE sobre un conjunto data (última columna = y). Si lineal, O=h; si no, usa beta y tanh."""
    x, y = _construir_x_con_bias(data)
    p = len(y)
    if p == 0:
        return 0.0
    if lineal:
        return calcular_error_mse_lineal(x, y, w, p)
    return _error_mse_no_lineal(x, y, w, p, beta, tanh)
