import os
import sys

# Permitir que las librerías utilicen múltiples núcleos para procesamiento de datos
os.environ["OMP_NUM_THREADS"] = "auto"
os.environ["MKL_NUM_THREADS"] = "auto"
os.environ["OPENBLAS_NUM_THREADS"] = "auto"
os.environ["VECLIB_MAXIMUM_THREADS"] = "auto"
os.environ["NUMEXPR_NUM_THREADS"] = "auto"

# Silenciar también alertas innecesarias de TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["MALLOC_CONF"] = "background_thread:true"
# Now you can safely import your machine learning libraries
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras import layers, Sequential, optimizers, applications, Model


def construir_cnn(
    input_shape: tuple,
    n_clases: int,
    filters_list: list[int],
    kernel_sizes_list: list[int],
    conv_strides_list: list[int],
    conv_padding: str,
    pool_sizes_list: list[int],
    pool_strides_list: list[int],
    pool_padding: str,
    dense_units: list[int],
    optimizer: str,
    learning_rate: float,
    loss: str,
) -> Sequential:
    """
    Construye la CNN secuencial con la arquitectura:
      Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → ...
      → Flatten → Dense → Dense → ... → Dense(N_CLASES, softmax)
    """
    n_conv = len(filters_list)

    # Auto-completar: si el usuario puso 3 filtros pero solo 2 strides,
    # se completa con el último valor para evitar IndexError.
    # Así FILTERS=[32,64,128] funciona aunque solo ponga CONV_STRIDES=[1,1].
    while len(kernel_sizes_list) < n_conv:
        kernel_sizes_list.append(kernel_sizes_list[-1] if kernel_sizes_list else 3)
    while len(conv_strides_list) < n_conv:
        conv_strides_list.append(conv_strides_list[-1] if conv_strides_list else 1)
    while len(pool_sizes_list) < n_conv:
        pool_sizes_list.append(pool_sizes_list[-1] if pool_sizes_list else 2)
    while len(pool_strides_list) < n_conv:
        pool_strides_list.append(pool_strides_list[-1] if pool_strides_list else 2)

    model = Sequential(name="CNN_Tp4")

    model.add(layers.Input(shape=input_shape))

    # --- Capas Conv2D + MaxPooling2D ---
    for i in range(n_conv):
        model.add(
            layers.Conv2D(
                filters=filters_list[i],
                kernel_size=kernel_sizes_list[i],
                strides=conv_strides_list[i],
                padding=conv_padding,
                activation="relu",
                name=f"conv_{i}",
            )
        )
        model.add(
            layers.MaxPooling2D(
                pool_size=pool_sizes_list[i],
                strides=pool_strides_list[i],
                padding=pool_padding,
                name=f"pool_{i}",
            )
        )

    # --- Aplanar ---
    model.add(layers.Flatten(name="flatten"))

    # --- Capas Dense ocultas ---
    for j, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{j}"))

    # --- Salida softmax ---
    model.add(layers.Dense(n_clases, activation="softmax", name="output"))

    # Compilar
    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    # Mostrar arquitectura
    print("\n" + "=" * 60)
    print("ARQUITECTURA CNN")
    print("=" * 60)
    model.summary()

    return model


def _get_optimizer(name: str, lr: float):
    opts = {
        "adam": optimizers.Adam(learning_rate=lr),
        "sgd": optimizers.SGD(learning_rate=lr),
        "rmsprop": optimizers.RMSprop(learning_rate=lr),
    }
    return opts.get(name.lower(), optimizers.Adam(learning_rate=lr))


# ═══════════════════════════════════════════════════════════════════
#  MODELO 2 · TRANSFER LEARNING
# ═══════════════════════════════════════════════════════════════════
# Diccionario con las bases preentrenadas disponibles. Cada una trae:
#   - la función que la construye (con pesos de ImageNet)
#   - su propio preprocess_input (cada red normaliza distinto la entrada)
#
#   "mobilenet"    → MobileNetV2. Liviana y rápida. Ideal para Mac / CPU.
#   "resnet"       → ResNet50.   Más pesada y precisa. Ideal con GPU NVIDIA.
#   "efficientnet" → EfficientNetB0. Buen equilibrio precisión/tamaño (GPU).
_BASES_PREENTRENADAS = {
    "mobilenet": (applications.MobileNetV2, applications.mobilenet_v2.preprocess_input),
    "resnet": (applications.ResNet50, applications.resnet50.preprocess_input),
    "efficientnet": (applications.EfficientNetB0, applications.efficientnet.preprocess_input),
}


def construir_transfer(
    input_shape: tuple,
    n_clases: int,
    base_name: str = "mobilenet",
    dense_units: list[int] | None = None,
    dropout: float = 0.3,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    loss: str = "categorical_crossentropy",
):
    """
    Construye el Modelo 2 con Transfer Learning:

        entrada → preprocess_input → base preentrenada (CONGELADA)
        → GlobalAveragePooling → Dense(ReLU) + Dropout → Dense(softmax)

    La base viene entrenada con ImageNet (millones de imágenes), así que ya
    "sabe mirar". La congelamos (base.trainable = False) y sólo entrenamos la
    cabeza nueva, adaptada a nuestras clases.

    Devuelve (model, base) — devolvemos también la base para poder
    descongelarla después en el fine-tuning.
    """
    if dense_units is None:
        dense_units = [128]

    base_name = base_name.lower()
    if base_name not in _BASES_PREENTRENADAS:
        print(f"[AVISO] Base '{base_name}' desconocida. Se usa 'mobilenet'.")
        base_name = "mobilenet"
    base_fn, preprocess = _BASES_PREENTRENADAS[base_name]

    # 1. Base preentrenada, sin la capa de clasificación de ImageNet
    base = base_fn(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # ← CONGELADA: no se re-entrena

    # 2. Armado funcional: entrada → preprocesado propio → base → cabeza nueva
    inputs = layers.Input(shape=input_shape)
    x = preprocess(inputs)  # cada modelo normaliza a su manera (espera 0-255)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    for u in dense_units:
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_clases, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name=f"transfer_{base_name}")

    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    print("\n" + "=" * 60)
    print(f"MODELO 2 · TRANSFER LEARNING  (base: {base_name})")
    print("=" * 60)
    print(f"  Base congelada: {len(base.layers)} capas (no se entrenan)")
    print(f"  Cabeza nueva  : GAP → densas {dense_units} → softmax({n_clases})")
    model.summary()

    return model, base


def fine_tuning(
    model,
    base,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs: int = 5,
    batch_size: int = 16,
    unfreeze: int = 30,
    learning_rate: float = 1e-5,
):
    """
    Ajuste fino (opcional). Descongela las ÚLTIMAS `unfreeze` capas de la base
    y las entrena con un learning rate muy chico, para afinar la red a nuestras
    imágenes sin romper lo que ya sabía. Se hace DESPUÉS de entrenar la cabeza.
    """
    print("\n" + "=" * 60)
    print(f"FINE-TUNING · descongelando las últimas {unfreeze} capas de la base")
    print("=" * 60)

    base.trainable = True
    # Congelar todas menos las últimas `unfreeze` capas
    for capa in base.layers[:-unfreeze]:
        capa.trainable = False

    # Recompilar (obligatorio tras cambiar trainable) con LR chico
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1,
    )


def entrenar(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO: {epochs} épocas, batch={batch_size}")
    print(f"{'='*60}\n")

    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1,
    )


def evaluar(model, x_test, y_test, clases, output_dir=None, sufijo=""):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print("EVALUACIÓN")
    print(f"{'='*60}")
    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")

    # Predicciones
    y_pred = model.predict(x_test, verbose=0)
    y_pred_c = np.argmax(y_pred, axis=1)
    y_true_c = np.argmax(y_test, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_true_c, y_pred_c)

    # Totales reales por fila para mostrarlos en las etiquetas
    # (así se ve cuántos había de cada clase, ej: "Gatos (19)")
    totales_fila = cm.sum(axis=1)
    etiq_reales = [f"{c}\n({totales_fila[i]})" for i, c in enumerate(clases)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clases,
        yticklabels=etiq_reales,
        annot_kws={"size": 14},
    )
    plt.title(f"Matriz de Confusión  (Acc: {acc*100:.1f}%)", fontsize=13)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.tight_layout()

    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ruta_guardado = os.path.join(output_dir, f"matriz_confusion{sufijo}.png")
    plt.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    print(f" Matriz de confusión guardada en: {ruta_guardado}")
    if plt.get_backend() != "agg":
        plt.show()

    print("\nReporte de Clasificación:")
    print(classification_report(y_true_c, y_pred_c, target_names=clases))

    return loss, acc


def mostrar_predicciones(
    model, x_test, y_test, clases, n_ejemplos=10, output_dir=None, sufijo=""
):
    y_pred = model.predict(x_test[:n_ejemplos], verbose=0)
    y_pred_c = np.argmax(y_pred, axis=1)
    y_true_c = np.argmax(y_test[:n_ejemplos], axis=1)

    n_cols = min(5, n_ejemplos)
    n_filas = (n_ejemplos + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 3, n_filas * 3.5))

    for i in range(n_ejemplos):
        plt.subplot(n_filas, n_cols, i + 1)

        # Mostrar la imagen (gris o color)
        img = x_test[i]
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(-1), cmap="gray")
        else:
            # Color: matplotlib espera float en 0-1. Si viene en 0-255
            # (caso transfer learning), lo escalamos sólo para mostrar.
            disp = img / 255.0 if img.max() > 1.0 else img
            plt.imshow(disp.astype("float32"))

        acerto = y_pred_c[i] == y_true_c[i]
        color = "green" if acerto else "red"
        marca = "✓" if acerto else "✗"
        titulo = f"Real: {clases[y_true_c[i]]}\nPred: {clases[y_pred_c[i]]} {marca}"
        plt.title(titulo, fontsize=10, color=color)
        plt.axis("off")

    plt.suptitle("Predicciones del modelo (10 ejemplos)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_guardado = os.path.join(output_dir, f"predicciones{sufijo}.png")
    plt.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    print(f"[INFO] Predicciones guardadas en: {ruta_guardado}")
    if plt.get_backend() != "agg":
        plt.show()


def predecir_imagen(ruta_imagen, model, clases, mostrar=True):
    """
    Inferencia sobre UNA imagen cualquiera (punto 5 de la consigna).

    Se le pasa la RUTA de cualquier imagen: la carga, la adapta a lo que espera
    el modelo (gris o color y el tamaño), la MUESTRA en pantalla y devuelve la
    clase predicha con su nivel de confianza.

    Funciona con los dos modelos sin cambiar nada: mira el input del modelo y
    solo detecta si necesita gris (Modelo 1) o color (Modelo 2, transfer).

    Ejemplo de uso:
        model = recuperar_modelo("tp_4/models", "transfer_mobilenet")
        predecir_imagen("mi_foto.jpg", model, CLASES)
    """
    from PIL import Image

    # ¿Qué espera el modelo? (alto, ancho, canales)
    _, alto, ancho, canales = model.input_shape
    modo = "RGB" if canales == 3 else "L"

    img = Image.open(ruta_imagen).convert(modo)
    img = img.resize((ancho, alto), Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32")

    if canales == 1:
        arr = arr / 255.0             # Modelo 1: normalizado a 0-1
        arr = np.expand_dims(arr, -1)  # (H, W) → (H, W, 1)
    # Modelo 2 (color): se deja en 0-255; el preprocess va adentro de la red.

    x = np.expand_dims(arr, 0)  # batch de 1 imagen: (1, H, W, C)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    clase = clases[idx]
    confianza = float(probs[idx]) * 100

    print(f"\n→ Predicción: {clase}  ({confianza:.1f}% de confianza)")

    if mostrar:
        disp = arr.squeeze()
        if canales == 3 and disp.max() > 1.0:
            disp = disp / 255.0  # sólo para mostrarla bien
        plt.figure(figsize=(4.5, 4.5))
        plt.imshow(disp, cmap="gray" if canales == 1 else None)
        plt.title(
            f"Predicción: {clase}  ({confianza:.1f}%)",
            fontsize=13,
            fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()
        if plt.get_backend() != "agg":
            plt.show()

    return clase, confianza


def guardar_modelo(model, directorio, nombre_base):
    os.makedirs(directorio, exist_ok=True)
    ruta = os.path.join(directorio, f"{nombre_base}.keras")
    model.save(ruta)
    print(f"\n Modelo guardado en: {ruta}")


def recuperar_modelo(directorio, nombre_base):
    ruta = os.path.join(directorio, f"{nombre_base}.keras")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta}")
    model = keras.models.load_model(ruta)
    print(f"\n Modelo recuperado desde: {ruta}")
    return model