import sys
import os
# .parents[1] apunta a TPs-RN, que es la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import keras
import matplotlib.pyplot as plt
from utils.helpers import (
    agregar_ruido, 
    read_json, 
    hex_a_binario, 
    prepare_items_for_keras, 
    display_pattern,
)
from tp_3.addons.functions import (
    autoencoder_keras_instance,
    graficar_espacio_latente,
    generar_caracter,
    punto_intermedio
)


def tp3():
    # --- CONFIGURACIÓN DE PRUEBA ---
    
    print("\n" + "="*50)
    print("INICIANDO TP3: AUTOENCODERS Y ESPACIO LATENTE")
    print("="*50)

    RECUPERAR_MODELO = True  # Cambiar a False para forzar un nuevo entrenamiento
    MODO_TANH = True
    output_act = "tanh" if MODO_TANH else "sigmoid"
    # tanh produce valores en (-1, +1): MSE es agnóstico al rango.
    # BCE requiere probabilidades (0, 1): incompatible con tanh.
    # MSE devuelve un escalar >= 0. Adam usa ∂MSE/∂W (puede ser + o -)
    # para actualizar cada W, permitiendo que W·h + b cubra (-∞, +∞)
    # antes de que tanh lo comprima a (-1, +1).
    loss_func = "mse" if MODO_TANH else "binary_crossentropy"
    threshold = 0.0 if MODO_TANH else 0.5
    path_sal = "./Documento/tp3.json"
    datos_sal = read_json(path_sal)
    matrix_binary = hex_a_binario(datos_sal)
    np_matrix = np.array(matrix_binary)
    x = np_matrix[:, :-1]
    y = np_matrix[:, -1]

    plane_data = prepare_items_for_keras(x)
    # Convertir a float32 para máxima compatibilidad con CUDA
    plane_data = np.array(plane_data).astype('float32') 

    if MODO_TANH:
        # Escalamos de [0, 1] a [-1, 1]
        plane_data = plane_data * 2 - 1

    print("Shape of plane_data:", plane_data.shape)

    # Verificación visual de los primeros 3 caracteres
    print("\n--- Verificación de datos de entrada ---")
    for i in range(3):
        # Para mostrar, volvemos a 0 y 1 si está en modo tanh
        vis_data = (plane_data[i] > threshold).astype(int)
        display_pattern(vis_data, label=y[i])

    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_files = ["autoencoder_completo.keras", "encoder.keras", "decoder.keras"]
    modelos_presentes = all(os.path.exists(os.path.join(models_path, f)) for f in model_files)

    if RECUPERAR_MODELO and modelos_presentes:
        print(f"\n[INFO] Recuperando modelos guardados desde: {models_path}")
        autoencoder = keras.models.load_model(os.path.join(models_path, "autoencoder_completo.keras"))
        encoder = keras.models.load_model(os.path.join(models_path, "encoder.keras"))
        decoder = keras.models.load_model(os.path.join(models_path, "decoder.keras"))
    else:
        if RECUPERAR_MODELO and not modelos_presentes:
            print(f"\n[AVISO] No se encontraron archivos de modelo. Iniciando entrenamiento...")

        autoencoder, encoder, decoder = autoencoder_keras_instance(output_activation=output_act, loss=loss_func)

        autoencoder.fit(
            plane_data, 
            plane_data, 
            epochs=2000,
            shuffle=True, 
            verbose=1,
            validation_split=0.1
        )
        
        # --- Exportar los modelos entrenados ---
        os.makedirs(models_path, exist_ok=True)
        autoencoder.save(os.path.join(models_path, "autoencoder_completo.keras"))
        encoder.save(os.path.join(models_path, "encoder.keras"))
        decoder.save(os.path.join(models_path, "decoder.keras"))
        print(f"\n[INFO] Modelos exportados exitosamente en: {models_path}")


    # --- FASE DE ANÁLISIS Y GENERACIÓN ---
    predict_autoencoder = autoencoder.predict(plane_data, verbose=0)
    
    # 1. Calcular puntos de interés antes de graficar
    punto_cero = [0.0, 0.0]
    # Mezcla del primer y segundo carácter del dataset (índices 0 y 1)
    mix_data = [3, 6]
    print("\n[PROCESO] punto intermedio latente...")
    for idx in mix_data:
        vis_data = (plane_data[idx] > threshold).astype(int)
        display_pattern(vis_data, label=y[idx])
    medio = punto_intermedio(encoder, plane_data, mix_data[0], mix_data[1])
    
    # 2. Gráfico del espacio latente incluyendo los puntos nuevos
    print("\n[PROCESO] Generando mapa del espacio latente...")
    puntos_interes = [punto_cero, medio]
    etiquetas_interes = ["Origen (0,0)", f"Mezcla ({mix_data[0]}-{mix_data[1]})"]
    graficar_espacio_latente(encoder, plane_data, y, 
                             puntos_extra=puntos_interes, 
                             etiquetas_extra=etiquetas_interes)

    # 3. Verificación visual de Reconstrucción
    print("\n" + "="*40)
    print("RECONSTRUCCIÓN DE PATRONES")
    print("="*40)

    reconstruccion_binaria = (predict_autoencoder > threshold).astype(int)
    original_binario = (plane_data > threshold).astype(int)

    for i in range(5):
        print(f"\nPatrón: '{y[i]}'")
        print("ORIGINAL:           RECONSTRUCCIÓN:")
        orig = display_pattern(original_binario[i], rows=7, cols=5, return_str=True)
        pred = display_pattern(reconstruccion_binaria[i], rows=7, cols=5, return_str=True)
        
        # Imprimir lado a lado
        for line_orig, line_pred in zip(orig.split('\n'), pred.split('\n')):
            print(f"{line_orig}       {line_pred}")

    # 4. Generación de caracteres en los puntos identificados
    print("\n" + "="*40)
    print("GENERACIÓN DE CARACTERES INÉDITOS")
    print("="*40)
    generar_caracter(decoder, punto_cero, threshold=threshold, path="./Documento/tp3_nuevo.png")
    generar_caracter(decoder, medio, threshold=threshold, path="./Documento/tp3_nuevo_mezcla.png")

    niveles_ruido = {
        "leve": 0.02,
        "moderado": 0.10,
        "severo": 0.25,
        "extremo": 0.50,
    }


    for nombre, prob in niveles_ruido.items():
        datos_ruidosos = []
        for patron in plane_data:
            patron_ruidoso = agregar_ruido(patron, prob=prob)
            datos_ruidosos.append(patron_ruidoso)
        datos_ruidosos = np.array(datos_ruidosos, dtype='float32')


        reconstruido = autoencoder.predict(datos_ruidosos, verbose=0)

        # se pasan valores a 0(-1 para tanh) y 1, segun el threshold
        ruidoso_bin = (datos_ruidosos > threshold).astype(int)
        reconstruido_bin = (reconstruido   > threshold).astype(int)
        
        # cuántos bits difieren entre ruidoso y original (cuánto ruido entró)
        bits_corrompidos = 0
        for i in range(len(original_binario)):
            for j in range(len(original_binario[i])):
                if ruidoso_bin[i][j] != original_binario[i][j]:
                    bits_corrompidos += 1
        total_bits = len(original_binario) * len(original_binario[0])
        tasa_ruido = bits_corrompidos / total_bits

        # cuántos bits difieren entre reconstruido y original (error residual)
        bits_error = 0
        for i in range(len(original_binario)):
            for j in range(len(original_binario[i])):
                if reconstruido_bin[i][j] != original_binario[i][j]:
                    bits_error += 1
        tasa_error = bits_error / total_bits

        print(f"\n[{nombre.upper():10s}] prob={prob}")
        print(f"  Ruido introducido : {tasa_ruido:.3f} ({bits_corrompidos}/{total_bits} bits)")
        print(f"  Error residual    : {tasa_error:.3f} ({bits_error}/{total_bits} bits)")


        IDX_EJEMPLO = 1
        print(f"\n  Ejemplo con '{y[IDX_EJEMPLO]}':")
        print(f"  {'ORIGINAL':<13}  {'RUIDOSO':<13}  RECONSTRUIDO")

        orig_lines  = display_pattern(original_binario[IDX_EJEMPLO],  rows=7, cols=5, return_str=True).split('\n')
        ruid_lines  = display_pattern(ruidoso_bin[IDX_EJEMPLO],       rows=7, cols=5, return_str=True).split('\n')
        recon_lines = display_pattern(reconstruido_bin[IDX_EJEMPLO],  rows=7, cols=5, return_str=True).split('\n')

        cantidad = min(len(orig_lines), len(ruid_lines), len(recon_lines))
        for i in range(cantidad):
            lo   = orig_lines[i]  if i < len(orig_lines)  else ""
            lr   = ruid_lines[i]  if i < len(ruid_lines)  else ""
            lrec = recon_lines[i] if i < len(recon_lines) else ""
            print(f"  {lo:<13}  {lr:<13}  {lrec}")
    return {}


# TODO: poner print para el conj de prueba
