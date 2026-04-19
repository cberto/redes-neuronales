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
def perceptron_simple(data, n, COTA, b, tanh):
    data = np.asanyarray(data)
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
        O = activacion(h, b, tanh)

        # Update the entire weight vector, not just one index
        dw = delta_w(n, y_μ, O, x_μ, h, b, tanh)

        # Sumamos el vector delta al vector de pesos elemento por elemento
        for w_i in range(len(w)):
            w[w_i] = w[w_i] + dw[w_i]

        error = calcular_error(x, y, w, p, b, tanh)
        if error < error_min:
            error_min = error
            w_min = w.copy()
        iterations += 1
    return w_min, error_min, iterations


def evaluar_perceptron_simple(x, w, b=None, tanh=False):
    x_μ = np.insert(x, 0, 1)  # agrega un 1 al inicio, para hacer x0 * w0
    h = excitacion(x_μ, w)
    O = activacion(h, b, tanh)
    return O


def calcular_error(x, y, w, p, b=None, tanh=False):
    total_error = 0
    for pos in range(p):
        x_μ = x[pos]
        y_μ = y[pos]
        h = excitacion(x_μ, w)  # calcular excitacion para el patron i
        O = activacion(h, b, tanh)
        total_error += (y_μ - O) ** 2
    return 0.5 * total_error


# g′(h) = β(1 − g(h)^2)
def sigmoide_tanh_d(b, h):
    g_h = sigmoide_tanh(b, h)
    return b * (1 - g_h**2)


# g(h) = tanh(βh)
def sigmoide_tanh(b, h):
    return Math.tanh(b * h)


# g′(h) = 2βg(h)(1 − g(h))
def sigmoidea_logica_d(b, h):
    g_h = sigmoidea_logica(b, h)
    return 2 * b * g_h * (1 - g_h)


# g(h) = 1 / (1 + e^(-2βh))
def sigmoidea_logica(b, h):
    return 1 / (1 + Math.exp(-2 * b * h))


def delta_w(n, y_μ, O, x_μ, h, b=None, tanh=False):
    g_h_d = 1
    if b:
        g_h_d = sigmoide_tanh_d(b, h) if tanh else sigmoidea_logica_d(b, h)

    factor_aprendizaje = n * (y_μ - O) * g_h_d

    # Creamos un vector vacío para el resultado
    dw = np.zeros(len(x_μ))

    # Multiplicamos el factor por cada componente de la entrada x_μ
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


def activacion(h, b=None, tanh_mode=False):
    """Activación según el tipo de perceptrón."""
    # Si hay parámetro b, usamos la función no lineal continua
    if b is not None:
        if tanh_mode:
            return sigmoide_tanh(b, h)
        else:
            return sigmoidea_logica(b, h)

    # Si no hay b, es el perceptrón simple original (función escalón)
    # Si h es positivo, la neurona se activa (1)
    if h > 0:
        return 1.0
    # Si h es negativo, la neurona se inhibe (-1)
    elif h < 0:
        return -1.0
    # Si es exactamente 0, se queda neutral
    return 0.0
