import os
import sys

# Permitir que las librerías utilicen múltiples núcleos para procesamiento de datos
os.environ["OMP_NUM_THREADS"] = "auto"
os.environ["MKL_NUM_THREADS"] = "auto"
os.environ["OPENBLAS_NUM_THREADS"] = "auto"
os.environ["VECLIB_MAXIMUM_THREADS"] = "auto"
os.environ["NUMEXPR_NUM_THREADS"] = "auto"

# Silenciar también alertas innecesarias de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" 
os.environ["MALLOC_CONF"] = "background_thread:true"
# Now you can safely import your machine learning libraries
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, Model


def autoencoder_keras_instance(dim_entrada=35, dim_latente=2, output_activation="sigmoid", loss="binary_crossentropy"):
    # Detectar entorno de forma ultra-robusta
    is_linux = sys.platform == 'linux'
    is_wsl = False
    if is_linux:
        try:
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    is_wsl = 'microsoft' in f.read().lower()
            else:
                is_wsl = os.path.exists('/dev/lxss') # Alternativa para algunos entornos WSL
        except:
            is_wsl = False

    # Configuración de GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    print("\n" + "="*50)
    print("DIAGNÓSTICO DE HARDWARE")
    print(f"Versión de TensorFlow: {tf.__version__}")
    print(f"Entorno: {'WSL2 (Linux) 🐧' if is_wsl else 'Windows Nativo 🪟'}")

    if gpus:
        try:
            # Esto evita que TF reserve toda la memoria de la GPU de golpe
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"¡ÉXITO! TensorFlow tiene acceso a la GPU.")
            print(f"Dispositivos: {len(gpus)} físicos, {len(logical_gpus)} lógicos")
            # Intenta obtener el nombre de la placa
            details = tf.config.experimental.get_device_details(gpus[0])
            print(f"Modelo: {details.get('device_name', 'Desconocido')}")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("ESTADO: ⚠️ Usando CPU 🐢")
        if not is_wsl:
            print("\nSUGERENCIA: Estás en Windows Nativo. TensorFlow >= 2.11 no soporta GPU aquí.")
            print("Usa el terminal de WSL2 (Ubuntu) para aprovechar tu RTX 5070.")
        else:
            py_ver = sys.version_info
            if py_ver.major == 3 and py_ver.minor >= 13:
                print(f"⚠️  ADVERTENCIA: Python {py_ver.major}.{py_ver.minor} es muy reciente.")
                print("   TensorFlow suele ser más estable en Python 3.10, 3.11 o 3.12.")
            
            print(f"Versión de Python en WSL2: {sys.version.split()[0]}")
            print("\nERROR: Estás en WSL2 pero la GPU no es visible.")
            
            # Intento de diagnóstico del sistema
            smi_check = os.system("nvidia-smi > /dev/null 2>&1")
            if smi_check != 0:
                print("❌ EL SISTEMA NO VE LA PLACA: Ejecuta 'nvidia-smi' en la terminal.")
                print("   Si falla, necesitas actualizar los drivers de NVIDIA en Windows.")
            else:
                print("✅ EL SISTEMA VE LA PLACA: Pero TensorFlow no.")
                print("   Esto significa que falta instalar las librerías de soporte:")
                print("   >>> pip install tensorflow[and-cuda]")
                print("   Asegúrate de estar dentro del venv_linux.")
                print("\n   TIP: Si ya lo instalaste, prueba ejecutar esto antes de correr el script:")
                print(f"   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python{py_ver.major}.{py_ver.minor}/site-packages/nvidia/cudnn/lib")

    print("="*50 + "\n")

    # 2. Construir el ENCODER
    inputs = layers.Input(shape=(dim_entrada,))
    
    # Si la entrada es grande (ej. 100x100=10000), necesitamos más capas
    if dim_entrada > 1000:
        x = layers.Dense(512, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
    else:
        x = layers.Dense(17, activation="relu")(inputs)
        
    latent_space = layers.Dense(dim_latente, activation="linear")(x)

    encoder = Model(inputs, latent_space, name="Encoder")

    # 3. Construir el DECODER
    decoder_inputs = layers.Input(shape=(dim_latente,))
    if dim_entrada > 1000:
        x = layers.Dense(128, activation="relu")(decoder_inputs)
        x = layers.Dense(512, activation="relu")(x)
    else:
        x = layers.Dense(17, activation="relu")(decoder_inputs)
        
    outputs = layers.Dense(dim_entrada, activation=output_activation)(x)

    decoder = Model(decoder_inputs, outputs, name="Decoder")

    # 4. Instanciar el AUTOENCODER completo
    autoencoder_outputs = decoder(encoder(inputs))
    autoencoder = Model(inputs, autoencoder_outputs, name="Autoencoder_Completo")

    # 5. Compilar el modelo
    autoencoder.compile(optimizer="adam", loss=loss)

    # 6. Mostrar el resumen en la terminal
    autoencoder.summary()

    return autoencoder, encoder, decoder


# =====================================================================
# PUNTO 3: gráfico en 2D del espacio latente
# =====================================================================
def graficar_espacio_latente(encoder, plane_data, etiquetas, 
                             puntos_extra=None, etiquetas_extra=None,
                             path="./Documento/tp3_latente.png"):
    """Pasa cada carácter por el encoder para obtener sus 2 coordenadas
    latentes (z1, z2) y las dibuja en el plano. Cada punto lleva al lado
    el carácter al que corresponde, así se ve cómo la red 'ordena' los
    caracteres parecidos cerca entre sí.
    """
    coords = encoder.predict(plane_data, verbose=0)  # forma (32, 2)

    plt.figure(figsize=(9, 8))
    # Dibujar puntos del dataset original
    plt.scatter(coords[:, 0], coords[:, 1], s=25, color="crimson", alpha=0.6, label="Dataset", zorder=3)
    for (z1, z2), c in zip(coords, etiquetas):
        plt.annotate(str(c), (z1, z2), fontsize=10, alpha=0.7,
                     xytext=(4, 4), textcoords="offset points")

    # Dibujar puntos generados o de interés extra
    if puntos_extra is not None:
        puntos_extra = np.array(puntos_extra)
        plt.scatter(puntos_extra[:, 0], puntos_extra[:, 1], s=100, color="blue", marker="X", label="Puntos de Interés", zorder=5)
        if etiquetas_extra:
            for (z1, z2), txt in zip(puntos_extra, etiquetas_extra):
                plt.annotate(txt, (z1, z2), fontsize=11, fontweight="bold", color="blue",
                             xytext=(6, 6), textcoords="offset points")

    plt.title("Mapeo de Caracteres en el Espacio Latente (2D)")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Gráfico del espacio latente guardado en: {path}")
    return coords


# =====================================================================
# PUNTO 4: generar un carácter NUEVO (que no está en el entrenamiento)
# =====================================================================
def generar_caracter(decoder, punto_latente, threshold=0.5, path="./Documento/tp3_nuevo.png"):
    """El decoder toma cualquier punto (z1, z2) del plano y reconstruye un
    carácter de 35 píxeles. Si el punto NO coincide con el de ningún
    carácter entrenado, la red está GENERANDO algo nuevo.

    Teoría: el autoencoder aprende un espacio latente continuo. Eso quiere
    decir que entre dos caracteres conocidos hay infinitos puntos válidos,
    y cada uno produce una imagen. Eligiendo un punto 'en el medio' (o uno
    cualquiera que no sea de los originales) obtenemos un carácter inédito.

    punto_latente: [z1, z2]
    """
    z = np.array([punto_latente], dtype="float32")
    salida = decoder.predict(z, verbose=0)[0]          # 35 valores entre 0 y 1
    imagen = (salida > threshold).astype(int)          # umbral dinámico

    print(f"\n Carácter generado en z=({punto_latente[0]:.2f}, "
          f"{punto_latente[1]:.2f}):")
    for i in range(7):
        fila = imagen[i * 5:(i + 1) * 5]
        print("".join("█" if b else " " for b in fila))

    plt.figure(figsize=(3, 4))
    plt.imshow(imagen.reshape(7, 5), cmap="binary")
    plt.title(f"Carácter nuevo\nz=({punto_latente[0]:.2f}, {punto_latente[1]:.2f})")
    plt.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Imagen del carácter nuevo guardada en: {path}")
    return imagen


def punto_intermedio(encoder, plane_data, idx_a, idx_b):
    """Devuelve el punto latente que está justo en el medio entre dos
    caracteres del dataset (índices idx_a e idx_b). Sirve para generar un
    carácter nuevo 'mezcla' de esos dos.
    """
    coords = encoder.predict(plane_data, verbose=0)
    medio = (coords[idx_a] + coords[idx_b]) / 2.0
    return medio.tolist()

