import os
import sys
from utils.helpers import like_json
from tp_1.part_1 import tp1
from tp_2.part_1 import tp2
from tp_3.part_1 import tp3
from tp_3.part_2 import tp3_part_3

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def main():
    #result_1 = tp1()
    #result_2 = tp2()
    result_3 = tp3_part_3()
    # result_3_2 = tp3_high_res()
    #print("Resultado 1_2: ", like_json(result_1))
    #print("Resultado 1_2: ", like_json(result_2))
    print("Resultado 1_3: ", like_json(result_3))
    



if __name__ == "__main__":
    main()
