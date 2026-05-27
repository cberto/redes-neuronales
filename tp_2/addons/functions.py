# # Algoritmo

# 1. Inicializar el conjunto de pesos en valores “pequeños” al azar.

# 2. Tomar un ejemplo \(\xi^\mu\) al azar del conjunto de entrenamiento y aplicarlo a la capa 0:

# \[
# V_i^0 = \xi_i^\mu \quad \text{para todo } i
# \]

# 3. Propagar la entrada hasta la capa de salida:

# \[
# V_i^m = g(h_i^m) = g\left(\sum_j w_{ij}^m V_j^{m-1}\right)
# \quad \text{para todo } m \text{ desde } 1 \text{ hasta } M
# \]

# 4. Calcular \(\delta\) para la capa de salida:

# \[
# \delta_i^M = g'(h_i^M)(\zeta_i^\mu - V_i^M)
# \]

# 5. Retropropagar \(\delta_i^m\) a la capa \(m-1\):

# \[
# \delta_i^{m-1} =
# g'(h_i^{m-1})
# \sum_j w_{ij}^m \delta_j^m
# \quad \text{para todo } m \text{ entre } M \text{ y } 2
# \]

# 6. Actualizar los pesos de las conexiones de acuerdo:

# \[
# w_{ij}^{\text{nuevo } m}
# =
# w_{ij}^{\text{viejo } m}
# +
# \Delta w_{ij}^m
# \]

# donde

# \[
# \Delta w_{ij}^m =
# \eta \, \delta_i^m \, V_j^{m-1}
# \]

# 7. Calcular el error.
# Si \(error > COTA\), ir a 2.
import numpy as np
import math as Math

# x = {{-1, 1}, {1, -1}, {-1, -1}, {1, 1}},
# y salida esperada y = {1, 1, -1, -1}.

layers = [2, 2, 2]


# COTA es hiperparametro de cantidad de iteraciones, p es cantidad de patrones, n es tasa de aprendizaje
def perceptron_multicapa(data, n, COTA, β, layers=[], tanh_mode=False):

    data = np.asanyarray(data)
    iterations = 0
    p = len(data)

    num_outputs = layers[-1]
    y = data[:, -num_outputs:]


    x_raw = data[:, :-num_outputs]
    x = np.ones((p, x_raw.shape[1] + 1))
    x[:, 1:] = x_raw

    weights = []
    # 1. Inicializar pesos: w_ij^m (valores pequeños al azar)
    for i in range(len(layers) - 1):
        # Matriz de (nodos siguiente) x (w0 + nodos actuales)
        shape = (layers[i + 1], layers[i] + 1)
        w_m = np.random.uniform(-0.1, 0.1, shape) # Pesos ma pequeños
        weights.append(w_m)

    x_μ = None
    y_μ = None
    actual_layer = 0
    w_min = [w.copy() for w in weights]
    error_costo_total = 0
    error = float("inf")
    while error > 0 and iterations < COTA:
        # 2. Seleccionar un patron al azar: V_i^0 = ξ_i^μ
        if actual_layer == 0:
            pos = np.random.randint(0, p)
            x_μ = x[pos]
            y_μ = y[pos]

        # 3. Propagacion hacia adelante: V_i^m = g(h_i^m) = g(Σ w_ij^m * V_j^{m-1})
        V = [x_μ]  # Activaciones de cada capa (la 0 es la entrada)
        H = []  # Excitaciones (h) de cada capa

        for m in range(len(weights)):
            h_capa = []
            v_siguiente = []

            for neurona_pesos in weights[m]:
                # h_i^m = Σ w_ij^m * V_j^{m-1}
                h = excitacion(V[-1], neurona_pesos)
                h_capa.append(h)
                o = activacion(h, β, tanh_mode)
                v_siguiente.append(o)

            H.append(np.array(h_capa))

            # Si no es la ultima capa, agregamos el w0 (1.0) para la siguiente
            if m < len(weights) - 1:
                v_siguiente.insert(0, 1.0)

            V.append(np.array(v_siguiente))

        # Necesitamos una lista para guardar los deltas de cada capa
        deltas = [None] * len(weights)

        # 4. Calcular delta de la capa de salida (M): δ_i^M = g'(h_i^M) * (ζ_i^μ - V_i^M)
        g_out = (
            sigmoide_tanh_derivada(β, H[-1])
            if tanh_mode
            else sigmoidea_logica_derivada(β, H[-1])
        )
        deltas[-1] = g_out * (y_μ - V[-1])

        # 5. Retropropagar deltas a capas ocultas: δ_i^{m-1} = g'(h_i^{m-1}) * Σ_j w_ij^m * δ_j^m
        # Vamos desde la penúltima capa hacia atrás hasta la primera
        for m in reversed(range(len(weights) - 1)):
            # g'(h) de la capa actual
            g_m = (
                sigmoide_tanh_derivada(β, H[m])
                if tanh_mode
                else sigmoidea_logica_derivada(β, H[m])
            )

            # --- SUMATORIA: Σ_j w_ij^m * δ_j^m ---
            # Para cada neurona 'i' en la capa actual
            sumatorias = np.zeros(len(H[m]))
            for i in range(len(H[m])):
                suma = 0
                # Sumamos la influencia de cada neurona 'j' de la capa siguiente
                for j in range(len(deltas[m + 1])):
                    # weights[m+1][j][i+1] es el peso que conecta i con j
                    w_ij_m = weights[m + 1][j][i + 1]
                    # Usamos i+1 porque el índice 0 de la matriz de pesos es el bias
                    δ_j_m = deltas[m + 1][j]
                    suma += w_ij_m * δ_j_m
                sumatorias[i] = suma

            deltas[m] = g_m * sumatorias

        # 6. Actualizar pesos: Δw_ij^m = η * δ_i^m * V_j^{m-1}
        for m in range(len(weights)):
            # Para cada neurona 'i' en la capa siguiente (destino de la conexión)
            for i in range(len(deltas[m])):
                # Para cada valor 'j' que viene de la capa anterior (origen de la conexión)
                for j in range(len(V[m])):
                    δ_i_m = deltas[m][i]
                    V_j_m = V[m][j]
                    weights[m][i][j] += n * δ_i_m * V_j_m

        
        error_costo_total = error_costo(x, y, weights, β, tanh_mode)
        if error_costo_total <= error:
            error = error_costo_total
            w_min = [w.copy() for w in weights]
        iterations += 1

    return w_min, error, iterations


def evaluar_perceptron_multicapa(x, weights, num_outputs, β=None, tanh_mode=False):
    x = np.asanyarray(x)
    # Determinamos cuántas entradas espera la primera capa (menos el w0)
    num_inputs = weights[0].shape[1] - 1
    # Tomamos solo los bits de entrada, ignorando etiquetas si las hay
    x_raw = x[:num_inputs]
    v_actual = np.insert(x_raw, 0, 1.0)
    
    for m in range(len(weights)):
        v_siguiente = []
        for neurona_pesos in weights[m]:
            h = excitacion(v_actual, neurona_pesos)
            o = activacion(h, β, tanh_mode)
            v_siguiente.append(o)
        if m < len(weights) - 1:
            v_actual = np.insert(v_siguiente, 0, 1.0)
        else:
            v_actual = np.array(v_siguiente)
            
    return v_actual


def error_costo(x, y, weights, β, tanh_mode):
    suma_total = 0.0
    for μ in range(len(x)):
        v_actual = x[μ]
        for m in range(len(weights)):
            v_siguiente = []
            for neurona_pesos in weights[m]:
                h = excitacion(v_actual, neurona_pesos)
                o = activacion(h, β, tanh_mode)
                v_siguiente.append(o)

            v_actual = np.array(v_siguiente)
            if m < len(weights) - 1:
                v_actual = np.insert(v_actual, 0, 1.0)

        suma_total += np.sum((y[μ] - v_actual) ** 2)

    return 0.5 * suma_total



def calcular_error(x, y, w, p, β=None, tanh_mode=False):
    total_error = 0
    for pos in range(p):
        x_μ = x[pos]
        y_μ = y[pos]
        h = excitacion(x_μ, w)  # calcular excitacion para el patron i
        O = activacion(h, β, tanh_mode)
        total_error += (y_μ - O) ** 2
    return 0.5 * total_error


# g′(h) = β(1 − g(h)^2)
def sigmoide_tanh_derivada(β, h):
    g_h = sigmoide_tanh(β, h)
    return β * (1 - g_h**2)


# g(h) = tanh(βh)
def sigmoide_tanh(β, h):
    return np.tanh(β * h)


# g′(h) = 2βg(h)(1 − g(h))
def sigmoidea_logica_derivada(β, h):
    g_h = sigmoidea_logica(β, h)
    return 2 * β * g_h * (1 - g_h)


# g(h) = 1 / (1 + e^(-2βh))
def sigmoidea_logica(β, h):
    return 1 / (1 + np.exp(-2 * β * h))


def delta_w(n, y_μ, O, x_μ, h, β=None, tanh_mode=False):
    g_h_d = 1
    if β is not None:
        g_h_d = (
            sigmoide_tanh_derivada(β, h)
            if tanh_mode
            else sigmoidea_logica_derivada(β, h)
        )

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


def activacion(h, β=None, tanh_mode=False):
    return sigmoide_tanh(β, h) if tanh_mode else sigmoidea_logica(β, h)
