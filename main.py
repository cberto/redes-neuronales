import os
import sys
from utils.helpers import like_json
from tp_1.part_1 import tp1
from tp_1.part_2 import tp1_ej2


sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def main():
    result_1 = tp1()
    print("TP1 — parte lógica (ejemplo actual): ", like_json(result_1))
    result_2 = tp1_ej2()
    print("TP1 — ejercicio 2 (archivos .txt): ", like_json(result_2))



if __name__ == "__main__":
    main()
