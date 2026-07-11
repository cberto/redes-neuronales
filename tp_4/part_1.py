import sys
import os
import tensorflow as tf
import keras
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import cargar_imagenes, graficar_historial
from tp_4.addons.functions import (
    construir_cnn,
    recuperar_modelo,
    entrenar,
    guardar_modelo,
    evaluar,
    mostrar_predicciones,
    predecir_imagen,
)


def _configurar_gpu():
    """Configura TensorFlow para usar GPU en WSL/Linux. Imprime diagnóstico."""
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO DE GPU")
    print("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(
                f"  GPU detectada: {len(gpus)} física(s), {len(logical_gpus)} lógica(s)"
            )
            details = tf.config.experimental.get_device_details(gpus[0])
            print(f"  Modelo: {details.get('device_name', 'Desconocido')}")
            print("  Estado: Usando GPU ✓")
        except RuntimeError as e:
            print(f"  Error configurando GPU: {e}")
            print("  Estado: Usando CPU ✗")
    else:
        print("  Estado: No se detectó GPU. Usando CPU 🐢")

        # Diagnóstico adicional para WSL
        if sys.platform == "linux":
            try:
                with open("/proc/version", "r") as f:
                    es_wsl = "microsoft" in f.read().lower()
            except:
                es_wsl = False

            if es_wsl:
                print("\n  Estás en WSL2. Posibles soluciones:")
                print("  1. pip install tensorflow[and-cuda]")
                print("  2. Asegurate de estar en el venv correcto")
                print("  3. nvidia-smi (si falla, actualizar drivers en Windows)")
                print("  4. export LD_LIBRARY_PATH=... (ver mensaje de error)")

    print("=" * 60 + "\n")


def tp4(imagen_prueba="./tp_4/test/images.jpeg"):
    _configurar_gpu()

    CLASES = [
        "Gatos",
        "Aves",
        "Caballos",
        "Tortugas",
        "Conejos",
        "Hipopótamos",
        "Perros",
        "Pingüinos",
        "Serpientes"       
    ]
    IMG_SIZE = 128  # Las originales son 512x512, entran 8 gatos en 64
    MAX_POR_CLASE = 100  # Cuántas imágenes por clase
    VALIDATION_SPLIT = 0.2

    # ─── Capas Conv2D ─────────────────────────────────────────────
    # FILTERS: cantidad de filtros por capa.
    #   [32, 64] = 1ra capa aprende 32 patrones, 2da capa aprende 64.
    #   NO es un filtro de 32×64. Cada filtro individual es de
    #   KERNEL_SIZES × KERNEL_SIZES píxeles (ej: 3×3).
    #
    # KERNEL_SIZES: tamaño de la "ventanita" que convoluciona.
    #   kernel=2 → ventana 2×2 (muy chica, ve poco contexto).
    #   kernel=3 → ventana 3×3 (lo mínimo recomendado).
    #   kernel=5 → ventana 5×5 (capta bordes y formas más grandes).
    #   kernel=7 → ventana 7×7 (para imágenes grandes como 512×512).
    #
    # STRIDES: el "paso" al desplazar la ventana.
    #   strides=1 → la ventana avanza de a 1 píxel (explora todo).
    #   strides=2 → avanza de a 2 (submuestrea, reduce tamaño).
    #   En imágenes chicas (64×64) se usa strides=1 para no perder
    #   información antes de tiempo.
    #
    # ─── Capas MaxPooling2D ──────────────────────────────────────
    # POOL_SIZES: ventana de muestreo. pool_size=2 → mira 2×2 píxeles.
    # POOL_STRIDES: paso del pooling. strides=2 → reduce a la MITAD.
    #
    # ¿QUÉ HACE MAXPOOLING?
    #   Agarra cada ventana de POOL_SIZES × POOL_SIZES (ej: 2×2) y
    #   se queda ÚNICAMENTE con el valor MÁXIMO de esos 4 píxeles.
    #   Descarta los otros 3. Esto:
    #     1) Reduce la dimensión (64×64 → 32×32)
    #     2) Hace el modelo "invariante" a pequeñas traslaciones
    #     3) Reduce la carga computacional

    FILTERS = [32, 64, 128]  # 32 filtros en 1ra capa, 64 en 2da
    KERNEL_SIZES = [5, 3, 3]  # ventana 3×3 (estándar mínimo)
    CONV_STRIDES = [1, 1, 1]  # píxel por píxel (no submuestrear)
    CONV_PADDING = "valid"  # "same" conserva tamaño, "valid" lo reduce
    POOL_SIZES = [2, 2, 2]  # ventana 2×2 para el pooling
    POOL_STRIDES = [2, 2, 2]  # stride=2 → reduce a la mitad
    POOL_PADDING = "valid"  # "same" o "valid"

    # ─── Capas Dense ──────────────────────────────────────────────
    DENSE_UNITS = [256, 128]  # Neuronas de las capas ocultas

    # ─── Entrenamiento ────────────────────────────────────────────
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    OPTIMIZER = "adam"
    LOSS = "categorical_crossentropy"

    # ─── Modelo ───────────────────────────────────────────────────
    RECUPERAR = True  # True → carga modelo guardado
    NOMBRE_MODELO = "cnn_clasificador"

    # ─── Inferencia (punto 5) ─────────────────────────────────────
    # La imagen llega por parámetro desde main (así corren los dos modelos
    # con la misma foto). Si no se pasa, usa la de ./tp_4/test por defecto.
    IMAGEN_PRUEBA = imagen_prueba

    n_clases = len(CLASES)
    input_shape = (IMG_SIZE, IMG_SIZE, 1)  # Siempre escala de grises, 1 canal

    # 1. Cargar imágenes
    print("\n[1/5] CARGANDO IMÁGENES...")
    print(f"  Clases: {CLASES}")
    print(f"  Tamaño: {IMG_SIZE}x{IMG_SIZE}  |  Máx x clase: {MAX_POR_CLASE}")
    print(f"  Padding Conv2D: '{CONV_PADDING}'  |  Padding Pool: '{POOL_PADDING}'")

    (x_train, y_train), (x_val, y_val) = cargar_imagenes(
        data_dir="./datasets/converted",
        clases=CLASES,
        target_size=IMG_SIZE,
        max_por_clase=MAX_POR_CLASE,
        validation_split=VALIDATION_SPLIT,
    )

    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=n_clases)
    y_val = keras.utils.to_categorical(y_val, num_classes=n_clases)

    # 2. Construir, entrenar o recuperar modelo
    print("\n[2/5] CONSTRUYENDO / RECUPERANDO MODELO...")
    ruta_modelos = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    if RECUPERAR:
        # ── Cargar modelo guardado (ya entrenado) ────────────────
        try:
            model = recuperar_modelo(ruta_modelos, NOMBRE_MODELO)
            print("  → Usando modelo guardado (sin re-entrenar).")
            historia = None
        except FileNotFoundError:
            print("[AVISO] No se encontró modelo guardado. Se entrenará uno nuevo.")
            RECUPERAR = False  # fuerza entrenamiento

    if not RECUPERAR:
        # ── Construir + Entrenar + Guardar ───────────────────────
        print("\n[3/5] CONSTRUYENDO RED...")
        model = construir_cnn(
            input_shape,
            n_clases,
            FILTERS,
            KERNEL_SIZES,
            CONV_STRIDES,
            CONV_PADDING,
            POOL_SIZES,
            POOL_STRIDES,
            POOL_PADDING,
            DENSE_UNITS,
            OPTIMIZER,
            LEARNING_RATE,
            LOSS,
        )

        print("\n[4/5] ENTRENANDO...")
        historia = entrenar(model, x_train, y_train, x_val, y_val, EPOCHS, BATCH_SIZE)

        print("\n[5/5] GUARDANDO MODELO ENTRENADO...")
        guardar_modelo(model, ruta_modelos, NOMBRE_MODELO)

    # ── Evaluar y visualizar (siempre, pase por donde pase) ────
    print("\n" + "=" * 60)
    print("EVALUANDO MODELO...")
    print("=" * 60)

    evaluar(model, x_val, y_val, CLASES)

    # Mostrar 10 ejemplos con su clasificación real vs predicha
    mostrar_predicciones(model, x_val, y_val, CLASES, n_ejemplos=10)

    if historia is not None:
        graficar_historial(
            historia,
            guardar=os.path.join(os.path.dirname(__file__), "historial.png"),
        )
        acc = historia.history["val_accuracy"][-1]
        loss = historia.history["val_loss"][-1]
    else:
        # Si se recuperó, no hay historial
        loss_val, acc_val = model.evaluate(x_val, y_val, verbose=0)
        acc = acc_val
        loss = loss_val
    print("\n" + "=" * 70)
    print(f"  Accuracy (val) : {acc*100:.2f}%")
    print(f"  Loss (val)     : {loss:.4f}")
    print("=" * 70)

    # ── Inferencia sobre la imagen elegida arriba (punto 5) ──────────
    # predecir_imagen convierte la foto a gris sola (este modelo usa 1 canal),
    # la muestra en pantalla y escribe la clase predicha con su confianza.
    print("\n" + "=" * 60)
    print(f"INFERENCIA SOBRE UNA IMAGEN: {IMAGEN_PRUEBA}")
    print("=" * 60)
    predecir_imagen(IMAGEN_PRUEBA, model, CLASES)

    return {"model": model, "accuracy": acc, "loss": loss}
