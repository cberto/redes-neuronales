
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
def perceptron_simple(data, n, COTA):
    i = 0
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
    error_min = float('inf')
    w_min = w.copy()

    while error > 0 and i < COTA:
        x_i = np.random.randint(0, p)
        h = excitacion1(x[x_i], w)
        O = activacion1(h) 
        
        # Update the entire weight vector, not just one index
        dw = delta_w(n, y[x_i], O, x[x_i])
        
        # Sumamos el vector delta al vector de pesos elemento por elemento
        for j in range(len(w)):
            w[j] = w[j] + dw[j]

        error = calcular_error(x, y, w, p)
        if error < error_min:
            error_min = error
            w_min = w.copy()
        i += 1
    return w_min, error_min, i

def calcular_error(x, y, w, p): 
    total_error = 0
    for j in range(p):
        h = excitacion1(x[j], w) # calcular excitacion para el patron j
        O = activacion1(h)
        total_error += (y[j] - O) ** 2
    return 0.5 * total_error
    
def delta_w(n, y_i, O, x_i):
    """
    Calcula el ajuste de pesos de forma explícita.
    No es un producto de matrices, es un ESCALAR multiplicado por un VECTOR.
    """
    # 1. Calculamos el factor común (escalar): tasa de aprendizaje * error
    factor_aprendizaje = n * (y_i - O)
    
    # 2. Creamos un vector vacío para el resultado (mismo tamaño que las entradas)
    dw = np.zeros(len(x_i))
    
    # 3. Multiplicamos el factor por cada componente de la entrada x_i
    # Esto nos da el ajuste individual para cada peso w_j
    for j in range(len(x_i)):
        dw[j] = factor_aprendizaje * x_i[j]
        
    return dw

def activacion(h):
    return np.sign(h)

def excitacion1(x, w):
    """Excitación h = x · w para un solo patrón x (vector fila) y pesos w."""
    h = 0.0
    for i in range(len(x)):
        x_i = float(x[i])
        w_i = float(w[i])
        h += x_i * w_i
    return h


def activacion1(h):
    """Activación O = signo(h) para el perceptrón (±1, 0 si h == 0)."""
    # Si h es positivo, la neurona se activa (1)
    if h > 0:
        return 1.0
    # Si h es negativo, la neurona se inhibe (-1)
    elif h < 0:
        return -1.0
    # Si es exactamente 0, se queda neutral
    return 0.0
