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
from keras import layers, Sequential, optimizers


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


def evaluar(model, x_test, y_test, clases, output_dir=None):
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

    ruta_guardado = os.path.join(output_dir, "matriz_confusion.png")
    plt.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    print(f" Matriz de confusión guardada en: {ruta_guardado}")
    plt.show()

    print("\nReporte de Clasificación:")
    print(classification_report(y_true_c, y_pred_c, target_names=clases))

    return loss, acc


def mostrar_predicciones(model, x_test, y_test, clases, n_ejemplos=10, output_dir=None):
    y_pred = model.predict(x_test[:n_ejemplos], verbose=0)
    y_pred_c = np.argmax(y_pred, axis=1)
    y_true_c = np.argmax(y_test[:n_ejemplos], axis=1)

    n_cols = min(5, n_ejemplos)
    n_filas = (n_ejemplos + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 3, n_filas * 3.5))

    for i in range(n_ejemplos):
        plt.subplot(n_filas, n_cols, i + 1)

        # Mostrar la imagen (escala de grises)
        img = x_test[i]
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(-1), cmap="gray")
        else:
            plt.imshow(img)

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
    ruta_guardado = os.path.join(output_dir, "predicciones.png")
    plt.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    print(f"[INFO] Predicciones guardadas en: {ruta_guardado}")
    plt.show()


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