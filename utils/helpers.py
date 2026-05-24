import csv
import math
import json
from matplotlib.pylab import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def like_json(data):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    return json.dumps(data, indent=2, ensure_ascii=False, default=_default)


def accuracy_score(tp, tn, fp, fn):
    # marca la deteccion de positivos, pero no negativos
    correct = tp + tn
    total = tp + tn + fp + fn
    return correct / total


def specificity_score(tn, fp):
    # probabilidad de detectar un neg
    return tn / (tn + fp) if tn + fp != 0 else 0


def recall_score(tp, fn):
    # probabilidad de detectar un pos
    return tp / (tp + fn) if fn + tp != 0 else 0


def precision_score(tp, fp):
    # probabilidad de detectar correctamente casos pos
    return tp / (tp + fp) if fp + tp != 0 else 0


def balanced_accuracy_score(recall, specificity):
    # similar a accuracy pero con clases desbalanceadas
    return (recall + specificity) / 2


def f1_score(precision, recall):
    # medida de calidad mas general
    return (
        2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    )


def TPR_score(tp, fn):
    # tasa verdaderos positivos
    return recall_score(tp, fn)


def FPR_score(fp, tn):
    # tasa falsos positivos
    return fp / (fp + tn) if tn + fp != 0 else 0


def complement_specificity_score(specificity):
    return 1 - specificity


def umbral_equal(recall_array, specificity_array):
    return np.argmin(np.abs(recall_array - specificity_array))


def umbral_youden(recall_array, specificity_array):
    return np.argmax(recall_array + specificity_array - 1)


def umbral_closer_point(recall_array, specificity_array):
    return np.argmin(recall_array**2 + (specificity_array - 1) ** 2)


def roc_curve(data, key_class, condicion_cumplida, path_to_save):
    TPR_array = []
    FPR_array = []
    length = len(data)
    for umbral in range(length):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index, item in enumerate(data):
            if umbral <= index:
                if item[key_class] == condicion_cumplida:
                    fn += 1
                else:
                    tn += 1
            else:
                if item[key_class] == condicion_cumplida:
                    tp += 1
                else:
                    fp += 1
        TPR = TPR_score(tp, fn)
        FPR = FPR_score(fp, tn)
        TPR_array.append(TPR)
        FPR_array.append(FPR)

    # PLOT HACIENDO SUAVIZADO DE CURVA
    x = np.array(FPR_array)
    y = np.array(TPR_array)
    # Ordenar por x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x, idx = np.unique(x, return_index=True)
    y = y[idx]

    x_new = np.linspace(x.min(), x.max(), 500)
    f = make_interp_spline(x, y, k=2)  # cuadrática
    y_smooth = f(x_new)

    plt.plot(x_new, y_smooth)
    plt.scatter(x, y)

    # Agregar etiquetas a los ejes
    plt.xlabel("Tasa de Falsos Positivos (FPR)")  # Etiqueta para el eje X
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")  # Etiqueta para el eje Y

    # Guardar y mostrar el gráfico
    plt.savefig(path_to_save)
    plt.show()

    return np.trapezoid(TPR_array, FPR_array)


def cargar_csv(file_path, encoding="utf-8", delimiter=","):
    try:
        data = []
        with open(file_path, "r", encoding=encoding) as file:
            csv_reader = csv.DictReader(file, delimiter=delimiter)
            data = [row for row in csv_reader]
        return data
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return []


def save_json(file_path, data, encoding="utf-8"):
    try:
        with open(file_path, "w", encoding=encoding) as archivo:
            json.dump(data, archivo, indent=2, ensure_ascii=False)
        print(f"Datos guardados en {file_path}")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        return []


def split_test_data(data, shuffle_data=False, test_size=0.2):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    # Mezclar los datos para asegurar representatividad de clases
    if shuffle_data:
        np.random.shuffle(data)
    length = len(data)
    parts = math.floor(length * test_size)
    set_length = length // parts - 1
    test, train = [], []
    pos_to_take = 0
    mod = length % parts

    for i in range(parts):
        start = length // parts * i if i != 0 else 0
        end = length // parts * (i + 1)

        if end > length:
            end = length
        # start y end van cambiando el orden en el cual eligen en cada iteracion un dato para test
        set_data = data[start:end]

        # verifica que la posicion no sea mayor al largo del set
        if pos_to_take >= set_length:
            pos_to_take = 0
        value = set_data.pop(pos_to_take)

        test.append(value)
        train.extend(set_data)

        # caso donde tengo resto
        if i + 1 == parts and mod != 0:
            set_data = data[end:length]
            if pos_to_take > len(set_data):
                pos_to_take = 0
            value = set_data.pop(pos_to_take)
            test.append(value)
            train.extend(set_data)
        pos_to_take += 1

    return test, train


def confusion_matrix(
    data, concepto, prediction_column, condicion_cumplida, val_medio=0.5
):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for row in data:
        print(row[concepto], row[prediction_column])
        if row[concepto] == condicion_cumplida:
            if row[prediction_column] > val_medio:
                tp += 1
            else:
                fn += 1
        else:
            if row[prediction_column] > val_medio:
                fp += 1
            else:
                tn += 1
    matrix = []
    matrix.append([tn, fp])
    matrix.append([fn, tp])
    #          PREDICCIÓN
    #         | No | Si |
    # REAL No | TN | FP |
    #      Si | FN | TP |
    str_matrix = [str([tn, fp])] + [str([fn, tp])]

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}, matrix, str_matrix


def filter_data_prestamo(data, attrs, concepto, ages, between=False):
    result = []
    to_filter = []
    if between:
        to_filter = range(ages[0], ages[1] + 1)
    else:
        to_filter = ages
    for row in data:
        if int(row["Edad"]) in to_filter:
            personal_data = {}
            for attr in attrs:
                personal_data[attr] = row[attr]
            result.append(
                {
                    **personal_data,
                    concepto: row[concepto],
                }
            )
    return result


def filter_data_estudiantes(data, attrs, respuesta, classification=False):
    result = []
    for row in data:
        personal_data = {}
        for attr in attrs:
            personal_data[attr] = float(row[attr])
        result.append(
            {
                **personal_data,
                respuesta: row[respuesta] if classification else float(row[respuesta]),
            }
        )
    return result


def filter_data_wines(data, attrs, respuesta):
    result = []
    for row in data:
        if row[respuesta] != "6":
            personal_data = {}
            for attr in attrs:
                personal_data[attr] = float(row[attr])
            quality = ""
            response = float(row[respuesta])
            if response < 5.0:
                quality = "REGULAR"
            if response > 5.0:
                quality = "EXCELENTE"
            if response == 5.0:
                quality = "BUENO"
            result.append(
                {
                    **personal_data,
                    respuesta: quality,
                }
            )
    return result


def plot_precision_vs_tree_size(node_counts, train_accuracies, test_accuracies, path):

    plt.figure(figsize=(10, 6))

    sorted_indices = np.argsort(node_counts)
    sorted_nodes = [node_counts[i] for i in sorted_indices]
    sorted_train_acc = [train_accuracies[i] for i in sorted_indices]
    sorted_test_acc = [test_accuracies[i] for i in sorted_indices]

    plt.plot(
        sorted_nodes,
        sorted_train_acc,
        "o-",
        label="Entrenamiento",
        color="blue",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        sorted_nodes,
        sorted_test_acc,
        "s-",
        label="Prueba",
        color="red",
        linewidth=2,
        markersize=6,
    )

    plt.xlabel("Número de Nodos Internos del Árbol", fontsize=12)
    plt.ylabel("Precisión", fontsize=12)
    plt.title(
        "Precisión vs Tamaño del Árbol\n(Entrenamiento vs Prueba)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    all_accuracies = sorted_train_acc + sorted_test_acc
    plt.ylim(min(all_accuracies) - 0.05, min(1.0, max(all_accuracies) + 0.05))

    if len(sorted_nodes) > 0:
        min_nodes = min(node_counts)
        max_nodes = max(node_counts)

        plt.annotate(
            f"Min: {min_nodes} nodos",
            xy=(min_nodes, min(train_accuracies)),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=9,
        )

        plt.annotate(
            f"Max: {max_nodes} nodos",
            xy=(max_nodes, max(train_accuracies)),
            xytext=(10, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Gráfico guardado como: {path}")


def confusion_matrix_multiclase(data, real_col, pred_col):
    # Obtener todas las clases posibles
    clases = sorted(list(set(row[real_col] for row in data)))
    n = len(clases)

    # Crear matriz vacía n x n
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    # Crear índice para mapear clase -> posición
    class_index = {clase: i for i, clase in enumerate(clases)}

    # Contar ocurrencias
    for row in data:
        real = row[real_col]
        pred = row[pred_col]
        i = class_index[real]
        j = class_index[pred]
        matrix[i][j] += 1
    # Real son Filas y Pred son Cols
    # Devolver matriz y clases para interpretación
    return matrix


def calculate_metrics(matrix):
    matrix = np.array(matrix)
    # Suma de la diagonal principal (TP para todas las clases)
    tp = np.diag(matrix)
    # Suma de cada fila (TP + FN)
    fn = np.sum(matrix, axis=1) - tp
    # Suma de cada columna (TP + FP)
    fp = np.sum(matrix, axis=0) - tp
    # Total de instancias
    total = np.sum(matrix)

    # Accuracy
    accuracy = np.sum(tp) / total

    # Recall por clase
    recall = np.divide(
        tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
    )

    # Precision por clase
    precision = np.divide(
        tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
    )

    # F1-score por clase
    f1 = np.divide(
        2 * (precision * recall),
        (precision + recall),
        out=np.zeros_like(tp, dtype=float),
        where=(precision + recall) != 0,
    )

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def cargar_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read().splitlines()
        return data
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return []


def plot_perceptron(w, data, error_min, iterations):

    # --- Lógica de Graficado Mejorada ---
    plt.figure(figsize=(10, 7))

    # Separamos los puntos por clase para identificarlos visualmente
    clase_pos = data[data[:, -1] == 1]
    clase_neg = data[data[:, -1] == -1]

    plt.scatter(
        clase_pos[:, 0],
        clase_pos[:, 1],
        color="royalblue",
        marker="o",
        s=100,
        label="Clase 1 (Otorgado)",
        edgecolors="k",
    )
    plt.scatter(
        clase_neg[:, 0],
        clase_neg[:, 1],
        color="crimson",
        marker="x",
        s=100,
        label="Clase -1 (Rechazado)",
    )

    # Definimos el rango de los ejes basado en los datos de entrada
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1  # x1 min y max
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1  # x2 min y max

    # Muestreo de x1 para dibujar la recta de decisión
    X = np.linspace(x_min, x_max, 100)

    # Despeje de x2:
    # w1*x1 + w2*x2 + w0 = 0  =>  w2*x2 = -w1*x1 - w0  =>  x2 = (-w1*x1 - w0) / w2
    # Esto es equivalente a: y = mx + b, donde m = -w1/w2 y b = -w0/w2
    if w[2] != 0:  # Evitar división por cero si la recta es vertical
        m = -w[1] / w[2]
        b = -w[0] / w[2]
        Y = m * X + b
        plt.plot(
            X,
            Y,
            color="forestgreen",
            lw=2,
            linestyle="--",
            label="Hiperplano de decisión",
        )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axhline(0, color="black", linewidth=1, alpha=0.3)
    plt.axvline(0, color="black", linewidth=1, alpha=0.3)
    plt.xlabel("Característica X1")
    plt.ylabel("Característica X2")
    plt.title(
        f"Perceptrón Simple: Separación Lineal Alcanzada\n(Error: {error_min}, Iteraciones: {iterations})"
    )
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def join_group_of_lists(list, group_size):
    len_list = len(list) // group_size
    items = []
    item = []
    for j in range(len_list):
        len_item = len(list[j])
        for s in range(group_size):
            for i in range(len_item):
                value = list[j * group_size + s][i]
                item.append(value)
        items.append(item)
        item = []
    return items

def what_number_is(bits35, patrones_referencia):
    for i, patron in enumerate(patrones_referencia):
        if np.array_equal(bits35, patron):
            return i
    return -1
    