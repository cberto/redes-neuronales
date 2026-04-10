import os
import sys
from utils.helpers import like_json
from tp_1.part_1 import tp1


sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def main():
    result_1 = tp1()
    print("Resultado 1_2: ", like_json(result_1))



if __name__ == "__main__":
    main()
