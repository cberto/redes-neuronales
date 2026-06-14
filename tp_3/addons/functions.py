import os
import sys

# 1. ESTO TIENE QUE IR ANTES QUE CUALQUIER OTRA LÍNEA
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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


def instancia_keras():

    # 1. Definir dimensiones
    dim_entrada = 784  # Ejemplo: una imagen de MNIST aplanada (28x28)
    dim_latente = 32  # Tamaño del espacio comprimido (cuello de botella)

    # 2. Construir el ENCODER
    inputs = layers.Input(shape=(dim_entrada,))
    encoder_layer = layers.Dense(128, activation="relu")(inputs)
    latent_space = layers.Dense(dim_latente, activation="relu")(encoder_layer)

    encoder = Model(inputs, latent_space, name="Encoder")

    # 3. Construir el DECODER
    decoder_inputs = layers.Input(shape=(dim_latente,))
    decoder_layer = layers.Dense(128, activation="relu")(decoder_inputs)
    outputs = layers.Dense(dim_entrada, activation="sigmoid")(decoder_layer)

    decoder = Model(decoder_inputs, outputs, name="Decoder")

    # 4. Instanciar el AUTOENCODER completo
    autoencoder_outputs = decoder(encoder(inputs))
    autoencoder = Model(inputs, autoencoder_outputs, name="Autoencoder_Completo")

    # 5. Compilar el modelo
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    # 6. Mostrar el resumen en la terminal
    autoencoder.summary()
