import sys
import os
# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import cargar_txt, read_json, hex_a_binario, prepare_items_for_keras, display_pattern
from tp_3.addons.functions import (
    autoencoder_keras_instance,
    graficar_espacio_latente,
    generar_caracter,
    punto_intermedio,
)


def tp3():
    
    path_sal = "./Documento/tp3.json"
    datos_sal = read_json(path_sal)
    matrix_binary = hex_a_binario(datos_sal)
    np_matrix = np.array(matrix_binary)
    x = np_matrix[:, :-1]
    y = np_matrix[:, -1]

    plane_data = prepare_items_for_keras(x)
    # Convertir a float32 para máxima compatibilidad con CUDA
    plane_data = np.array(plane_data).astype('float32') 
    print("Shape of plane_data:", plane_data.shape)

    # Verificación visual de los primeros 3 caracteres
    print("\n--- Verificación de datos de entrada ---")
    for i in range(3):
        display_pattern(plane_data[i], label=y[i])

    autoencoder, encoder, decoder = autoencoder_keras_instance()
    # Aumentamos épocas: la RTX 5070 lo procesará casi instantáneamente
    # verbose=1 muestra la barra de progreso. 
    # validation_split=0.2 separa un 20% de los datos para evaluar el modelo en cada época.
    autoencoder.fit(
        plane_data, 
        plane_data, 
        epochs=2000, 
        shuffle=True, 
        verbose=1,
        validation_split=0.1
    )
    
    # --- Exportar los modelos entrenados ---
    # Creamos una carpeta 'models' dentro de tp_3 si no existe
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_path, exist_ok=True)

    autoencoder.save(os.path.join(models_path, "autoencoder_completo.keras"))
    encoder.save(os.path.join(models_path, "encoder.keras"))
    decoder.save(os.path.join(models_path, "decoder.keras"))
    print(f"\n[INFO] Modelos exportados exitosamente en: {models_path}")

    predict_autoencoder = autoencoder.predict(plane_data)
    
    # encoder usa los datos originales como entrada
    predict_encoder = encoder.predict(plane_data)

    # decoder usa los datos de encoder como datos de entrada
    predict_decoder = decoder.predict(predict_encoder)

    # Verificación visual de los resultados (Primeros 5 caracteres)
    print("\n" + "="*40)
    print("VERIFICACIÓN DE RECONSTRUCCIÓN")
    print("="*40)
    
    # Aplicamos un umbral de 0.5 para binarizar la salida del autoencoder
    reconstruccion_binaria = (predict_autoencoder > 0.5).astype(int)

    for i in range(5):
        print(f"\nCarácter: '{y[i]}'")
        print("ORIGINAL:           RECONSTRUCCIÓN:")
        orig = display_pattern(plane_data[i], rows=7, cols=5, return_str=True)
        pred = display_pattern(reconstruccion_binaria[i], rows=7, cols=5, return_str=True)
        
        # Imprimir lado a lado
        for line_orig, line_pred in zip(orig.split('\n'), pred.split('\n')):
            print(f"{line_orig}       {line_pred}")

    # --- PUNTO 3: gráfico del espacio latente en 2D ---
    graficar_espacio_latente(encoder, plane_data, y)

    # --- PUNTO 4: generar un carácter nuevo (no entrenado) ---
    # Opción A: un punto cualquiera del plano elegido a mano
    generar_caracter(decoder, [0.0, 0.0], path="./Documento/tp3_nuevo.png")

    # Opción B: el punto que está justo entre dos caracteres conocidos
    # (mezcla del primero y el segundo del dataset). Cambiá los índices
    # para probar otras combinaciones.
    medio = punto_intermedio(encoder, plane_data, 0, 1)
    generar_caracter(decoder, medio, path="./Documento/tp3_nuevo_mezcla.png")

    return {}


# TODO: poner print para el conj de prueba
