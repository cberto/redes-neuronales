import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from utils.helpers import fetch_pokemon_as_input, display_pattern
from tp_3.addons.functions import autoencoder_keras_instance

def tp3_part_3():
    print("\n--- EJECUTANDO PARTE 2: POKÉMON 100x100 ---")
    
    # 1. Parámetros de alta resolución
    IMG_SIZE = (100, 100)
    DIM_INPUT = IMG_SIZE[0] * IMG_SIZE[1]
    DIM_LATENT = 64  # Aumentamos el cuello de botella para capturar detalles
    
    # Traemos los primeros 151 Pokémon (Kanto)
    poke_ids = list(range(1, 152))
    print(f"[INFO] Descargando y procesando {len(poke_ids)} imágenes...")
    
    # Reutilizamos el helper con el nuevo tamaño
    poke_data, poke_names = fetch_pokemon_as_input(poke_ids, target_size=IMG_SIZE)
    
    # 2. Instanciar y Entrenar
    # Usamos MSE porque Tanh/Sigmoid en 10,000 neuronas requiere gradientes estables
    autoencoder, encoder, decoder = autoencoder_keras_instance(
        dim_entrada=DIM_INPUT, 
        dim_latente=DIM_LATENT,
        output_activation="sigmoid", # Usamos sigmoid 0-1 para imágenes
        loss="mse"
    )

    print("[INFO] Entrenando Autoencoder de Alta Resolución...")
    autoencoder.fit(
        poke_data, poke_data,
        epochs=100, # Menos épocas porque hay más parámetros, la GPU trabajará duro
        batch_size=16,
        shuffle=True,
        verbose=1,
        validation_split=0.1
    )

    # 3. Verificación
    idx = np.random.randint(0, len(poke_data))
    sample = poke_data[idx:idx+1]
    reconstruction = autoencoder.predict(sample, verbose=0)

    # Visualización
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {poke_names[idx]}")
    plt.imshow(sample.reshape(IMG_SIZE[1], IMG_SIZE[0]), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstrucción")
    plt.imshow(reconstruction.reshape(IMG_SIZE[1], IMG_SIZE[0]), cmap='gray')
    plt.axis('off')

    out_path = "./Documento/tp3_poke_100x100.png"
    plt.savefig(out_path)
    print(f"[INFO] Resultado guardado en {out_path}")
    plt.show()

    return {"status": "success", "processed": len(poke_ids)}