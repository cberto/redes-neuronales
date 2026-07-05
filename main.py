"""
Configuración para descargar y organizar imágenes desde Kaggle.
"""
from tp_2.part_1 import tp2
from tp_3.part_1 import tp3
from tp_3.part_2 import tp3_part_3
from tp_4.part_1 import tp4
from utils.preparar_datos import descargar
from utils.helpers import convert_dataset_img

API_COMMAND = "competitions download -c dogs-vs-cats-redux-kernels-edition"
DEST_DIR = "./datasets"
CATEGORIAS = ["cat", "dog"]
LIMITE = 100

def main():
    # ── Ejecutar TP4 (parámetros definidos dentro de tp4() en MAYÚSCULAS) ──
    resultados = tp4()
    # descargar(
    #         api_command=API_COMMAND,
    #         dest_dir=DEST_DIR,
    #         categorias=CATEGORIAS,
    #         limite=LIMITE,
    # )
    # convert_dataset_img(DEST_DIR, CATEGORIAS, [512, 512])
    # result_1 = tp1()
    # result_2 = tp2()
    # result_3 = tp3()
    # result_3_2 = tp3_part_3()
    # print("Resultado 1: ", like_json(result_1))
    # print("Resultado 2: ", like_json(result_2))
    # print("Resultado 3: ", like_json(result_3))

    # También puedes bajar el kernel de Kaggle (como antes)
    # command = "kaggle kernels pull gpreda/cats-or-dogs-using-cnn-with-transfer-learning"
    # get_data_kaggle(command, doc_path="./kaggle_data.json")

    # Ejecutar pipeline completo de Cats vs Dogs


if __name__ == "__main__":
    main()
