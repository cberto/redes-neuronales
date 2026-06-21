import os
import sys
from utils.helpers import like_json
from tp_1.part_1 import tp1
from tp_2.part_1 import tp2
from tp_3.part_1 import tp3
from tp_3.part_2 import tp3_part_3

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(line_buffering=True)  # 


def main():
    #result_1 = tp1()
    #result_2 = tp2()
    #result_3_1 = tp3()
    result_3_3 = tp3_part_3()
    #print("Resultado 1_2: ", like_json(result_1))
    #print("Resultado 1_2: ", like_json(result_2))
    #print("Resultado 1_3: ", like_json(result_3_1))
    #print("Resultado 1_3: ", like_json(result_3_3))
    



if __name__ == "__main__":
    main()
