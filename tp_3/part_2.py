import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from utils.helpers import fetch_pokemon_as_input, display_pattern
from tp_3.addons.functions import autoencoder_keras_instance


def tp3_part_3():
    print("\n--- EJECUTANDO PARTE 3: POKÉMON 100x100 ---")

    RECUPERAR_MODELO = True

    IMG_SIZE = (100, 100)
    DIM_INPUT = IMG_SIZE[0] * IMG_SIZE[1]
    DIM_LATENT = 64

    poke_ids = list(range(1, 152))

    print(f"Descargando y procesando {len(poke_ids)} imágenes...")

    poke_data, poke_names = fetch_pokemon_as_input(poke_ids, target_size=IMG_SIZE)

    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    os.makedirs(models_path, exist_ok=True)

    auto_path = os.path.join(models_path, "autoencoder_pokemon.keras")
    enc_path = os.path.join(models_path, "encoder_pokemon.keras")
    dec_path = os.path.join(models_path, "decoder_pokemon.keras")

    modelos_presentes = (
        os.path.exists(auto_path)
        and os.path.exists(enc_path)
        and os.path.exists(dec_path)
    )

    if RECUPERAR_MODELO and modelos_presentes:
        print("Recuperando modelos entrenados...")
        autoencoder = keras.models.load_model(auto_path)
        encoder = keras.models.load_model(enc_path)
        decoder = keras.models.load_model(dec_path)
    else:
        print("Creando modelo...")
        autoencoder, encoder, decoder = autoencoder_keras_instance(
            dim_entrada=DIM_INPUT,
            dim_latente=DIM_LATENT,
            output_activation="sigmoid",
            loss="mse",
        )
        print("Entrenando Autoencoder...")

        autoencoder.fit(
            poke_data,
            poke_data,
            epochs=5000,
            batch_size=16,
            shuffle=True,
            verbose=1,
            validation_split=0.1,
        )

        print("Guardando modelos...")
        autoencoder.save(auto_path)
        encoder.save(enc_path)
        decoder.save(dec_path)

    idx = np.random.randint(0, len(poke_data))
    sample = poke_data[idx : idx + 1]

    reconstruction = autoencoder.predict(sample, verbose=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {poke_names[idx]}")
    plt.imshow(sample.reshape(IMG_SIZE), cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Reconstrucción")
    plt.imshow(reconstruction.reshape(IMG_SIZE), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    out_path = "./Documento/tp3_poke_100x100.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Resultado guardado en {out_path}")

   
    # PUNTO ENTRE 2 PUNTOS CONOCIDOS DE LA DIMESION LATENTE

    idx_a = 3
    idx_b = 4

    poke_a = poke_data[idx_a : idx_a + 1]
    poke_b = poke_data[idx_b : idx_b + 1]

    z_a = encoder.predict(poke_a, verbose=0)
    z_b = encoder.predict(poke_b, verbose=0)

    pasos = 7

    fig, axes = plt.subplots(1, pasos, figsize=(18, 4))

    for i, t in enumerate(np.linspace(0, 1, pasos)):
        z = (1 - t) * z_a + t * z_b

        generated = decoder.predict(z, verbose=0)
        axes[i].imshow(generated.reshape(IMG_SIZE), cmap="gray")
        axes[i].set_title(f"t={t:.2f}")
        axes[i].axis("off")

    plt.suptitle("Interpolación en el espacio latente")
    plt.tight_layout()
    plt.savefig("./Documento/tp3_poke_interpolacion.png", dpi=150)
    plt.close()

    print("Interpolación latente guardada")

    return {"processed": len(poke_ids), "latent_dim": DIM_LATENT}
