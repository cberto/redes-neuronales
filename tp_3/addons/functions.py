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
from keras import layers, Model


def autoencoder_keras_instance():
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

    # 1. Definir dimensiones
    dim_entrada = 35  # Ejemplo: una imagen de MNIST aplanada (28x28)
    dim_latente = 2  # Tamaño del espacio comprimido (cuello de botella)

    # 2. Construir el ENCODER
    inputs = layers.Input(shape=(dim_entrada,))
    encoder_layer = layers.Dense(17, activation="relu")(inputs)
    latent_space = layers.Dense(dim_latente, activation="relu")(encoder_layer)

    encoder = Model(inputs, latent_space, name="Encoder")

    # 3. Construir el DECODER
    decoder_inputs = layers.Input(shape=(dim_latente,))
    decoder_layer = layers.Dense(17, activation="relu")(decoder_inputs)
    outputs = layers.Dense(dim_entrada, activation="sigmoid")(decoder_layer)

    decoder = Model(decoder_inputs, outputs, name="Decoder")

    # 4. Instanciar el AUTOENCODER completo
    autoencoder_outputs = decoder(encoder(inputs))
    autoencoder = Model(inputs, autoencoder_outputs, name="Autoencoder_Completo")

    # 5. Compilar el modelo
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    # 6. Mostrar el resumen en la terminal
    autoencoder.summary()

    return autoencoder, encoder, decoder