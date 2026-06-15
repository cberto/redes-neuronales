import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _ensure_venv():
    if sys.platform == "win32":
        venv_python = os.path.join(_ROOT, "venv_linux", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(_ROOT, "venv_linux", "bin", "python")

    if not os.path.isfile(venv_python):
        print("Entorno virtual no encontrado. Ejecutá:")
        print("  make -f Makefile-Macos init")
        sys.exit(1)

    if os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.execv(venv_python, [venv_python] + sys.argv)


_ensure_venv()
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, _ROOT)

from utils.helpers import like_json
from tp_1.part_1 import tp1
from tp_2.part_1 import tp2
from tp_3.part_1 import tp3
from tp_3.part_2 import tp3_part_3


def main():
    result_1 = tp1()
    # result_2 = tp2()
    # result_3 = tp3_part_3()
    # result_3_2 = tp3_high_res()
    print("Resultado 1_2: ", like_json(result_1))
    # print("Resultado 1_2: ", like_json(result_2))
    # print("Resultado 1_3: ", like_json(result_3))


if __name__ == "__main__":
    main()
