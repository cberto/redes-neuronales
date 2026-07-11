import sys
import os
import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import cargar_imagenes_color, graficar_historial
from tp_4.addons.functions import (
    construir_transfer,
    fine_tuning,
    recuperar_modelo,
    entrenar,
    guardar_modelo,
    evaluar,
    mostrar_predicciones,
    predecir_imagen,
)


def tp4_transfer(imagen_prueba="./tp_4/test/images.jpeg"):
    """
    Modelo 2 · Transfer Learning.

    Reutiliza una red preentrenada con ImageNet (MobileNetV2 / ResNet50 /
    EfficientNetB0), le congela la base y le entrena una cabeza nueva para
    NUESTRAS clases. Opcionalmente hace un fine-tuning al final.

    Comparte el mismo dataset y las mismas funciones de evaluación que el
    Modelo 1 (part_1.py); sólo cambia cómo se construye el modelo y que las
    imágenes se cargan en COLOR (los preentrenados esperan 3 canales).
    """

    # ─── Configuración ────────────────────────────────────────────────
    # Cuando subas las clases nuevas, agregá acá sus nombres (carpetas dentro
    # de datasets/converted). La consigna pide llegar a 8.
    CLASES = [
        "Gatos",
        "Aves",
        "Caballos",
        "Tortugas",
        "Conejos",
        "Hipopótamos",
        "Perros",
        "Pingüinos",
        "Serpientes",
    ]

    # ┌─────────────────────────────────────────────────────────────────┐
    # │  ELEGÍ LA BASE PREENTRENADA SEGÚN TU MÁQUINA:                    │
    # │    "mobilenet"    → liviana y rápida. Recomendada para Mac / CPU. │
    # │    "resnet"       → más precisa y pesada. Ideal con GPU NVIDIA.   │
    # │    "efficientnet" → equilibrio precisión/tamaño. Ideal con GPU.   │
    # └─────────────────────────────────────────────────────────────────┘
    MODELO_BASE = "mobilenet"

    IMG_SIZE = 128          # los preentrenados aceptan tamaños >= 32
    MAX_POR_CLASE = 100
    VALIDATION_SPLIT = 0.2

    # ─── Cabeza nueva (lo único que entrenamos al principio) ──────────
    DENSE_UNITS = [128]     # capas densas de la cabeza
    DROPOUT = 0.3           # apaga neuronas al azar → menos sobreajuste

    # ─── Entrenamiento de la cabeza ───────────────────────────────────
    EPOCHS = 15
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001

    # ─── Fine-tuning (opcional) ───────────────────────────────────────
    HACER_FINE_TUNING = True
    FT_UNFREEZE = 30        # cuántas capas finales de la base descongelar
    FT_EPOCHS = 5
    FT_LEARNING_RATE = 1e-5  # muy chico para no romper lo aprendido

    # ─── Modelo ───────────────────────────────────────────────────────
    RECUPERAR = True
    NOMBRE_MODELO = f"transfer_{MODELO_BASE}"

    # ─── Inferencia (punto 5) ─────────────────────────────────────────
    # La imagen llega por parámetro desde main (así corren los dos modelos
    # con la misma foto). Si no se pasa, usa la de ./tp_4/test por defecto.
    IMAGEN_PRUEBA = imagen_prueba

    n_clases = len(CLASES)
    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # COLOR: 3 canales

    # 1. Cargar imágenes EN COLOR y sin normalizar (0-255)
    print("\n[1/5] CARGANDO IMÁGENES (color, para transfer learning)...")
    print(f"  Clases: {CLASES}")
    print(f"  Base preentrenada: {MODELO_BASE}  |  Tamaño: {IMG_SIZE}x{IMG_SIZE}x3")

    (x_train, y_train), (x_val, y_val) = cargar_imagenes_color(
        data_dir="./datasets/converted",
        clases=CLASES,
        target_size=IMG_SIZE,
        max_por_clase=MAX_POR_CLASE,
        validation_split=VALIDATION_SPLIT,
    )

    y_train = keras.utils.to_categorical(y_train, num_classes=n_clases)
    y_val = keras.utils.to_categorical(y_val, num_classes=n_clases)

    # 2. Construir o recuperar
    print("\n[2/5] CONSTRUYENDO / RECUPERANDO MODELO...")
    ruta_modelos = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    historia = None
    historia_ft = None

    if RECUPERAR:
        try:
            model = recuperar_modelo(ruta_modelos, NOMBRE_MODELO)
            print("  → Usando modelo guardado (sin re-entrenar).")
        except FileNotFoundError:
            print("[AVISO] No se encontró modelo guardado. Se entrenará uno nuevo.")
            RECUPERAR = False

    if not RECUPERAR:
        # 3. Construir (base congelada + cabeza nueva)
        print("\n[3/5] CONSTRUYENDO RED (base congelada + cabeza nueva)...")
        model, base = construir_transfer(
            input_shape,
            n_clases,
            base_name=MODELO_BASE,
            dense_units=DENSE_UNITS,
            dropout=DROPOUT,
            learning_rate=LEARNING_RATE,
        )

        # 4. Entrenar SÓLO la cabeza
        print("\n[4/5] ENTRENANDO LA CABEZA (base congelada)...")
        historia = entrenar(
            model, x_train, y_train, x_val, y_val, EPOCHS, BATCH_SIZE
        )

        # 4b. Fine-tuning opcional
        if HACER_FINE_TUNING:
            historia_ft = fine_tuning(
                model, base, x_train, y_train, x_val, y_val,
                epochs=FT_EPOCHS,
                batch_size=BATCH_SIZE,
                unfreeze=FT_UNFREEZE,
                learning_rate=FT_LEARNING_RATE,
            )

        print("\n[5/5] GUARDANDO MODELO ENTRENADO...")
        guardar_modelo(model, ruta_modelos, NOMBRE_MODELO)

    # ── Evaluar y visualizar (archivos con sufijo para no pisar el Modelo 1) ──
    print("\n" + "=" * 60)
    print("EVALUANDO MODELO 2 (TRANSFER LEARNING)...")
    print("=" * 60)

    evaluar(model, x_val, y_val, CLASES, sufijo="_transfer")
    mostrar_predicciones(
        model, x_val, y_val, CLASES, n_ejemplos=10, sufijo="_transfer"
    )

    if historia is not None:
        # Unir el historial de la cabeza + el del fine-tuning para el gráfico
        hist = _unir_historiales(historia, historia_ft)
        graficar_historial(
            hist,
            guardar=os.path.join(os.path.dirname(__file__), "historial_transfer.png"),
        )
        acc = hist.history["val_accuracy"][-1]
        loss = hist.history["val_loss"][-1]
    else:
        loss_val, acc_val = model.evaluate(x_val, y_val, verbose=0)
        acc, loss = acc_val, loss_val

    print("\n" + "=" * 70)
    print(f"  MODELO 2 · {MODELO_BASE}")
    print(f"  Accuracy (val) : {acc*100:.2f}%")
    print(f"  Loss (val)     : {loss:.4f}")
    print("=" * 70)

    # ── Inferencia sobre la imagen elegida arriba (punto 5) ──────────
    # predecir_imagen usa la foto en color (este modelo espera 3 canales),
    # la muestra en pantalla y escribe la clase predicha con su confianza.
    print("\n" + "=" * 60)
    print(f"INFERENCIA SOBRE UNA IMAGEN: {IMAGEN_PRUEBA}")
    print("=" * 60)
    predecir_imagen(IMAGEN_PRUEBA, model, CLASES)

    return {"model": model, "base": MODELO_BASE, "accuracy": acc, "loss": loss}


class _Hist:
    """Envuelve un dict de historial para que graficar_historial lo acepte."""
    def __init__(self, history):
        self.history = history


def _unir_historiales(h1, h2=None):
    """Concatena el historial de la cabeza con el del fine-tuning (si hubo)."""
    claves = ["loss", "val_loss", "accuracy", "val_accuracy"]
    combinado = {k: list(h1.history.get(k, [])) for k in claves}
    if h2 is not None:
        for k in claves:
            combinado[k] += list(h2.history.get(k, []))
    return _Hist(combinado)


if __name__ == "__main__":
    tp4_transfer()
