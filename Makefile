# macOS / Linux (por defecto). En Windows: make -f Makefile.win init
# `make init` = venv + requirements-runtime.txt + ejecutar main.py.
# Dependencias fijadas del curso (kaggle, etc.): make deps-full
.PHONY: init install-venv install-deps deps-full run ej2 update-deps

init: install-venv install-deps run

PYTHON=venv/bin/python
PIP=venv/bin/pip

install-venv:
	@if [ -d venv ]; then \
		echo "venv ya existe."; \
	else \
		echo "Creando venv..."; \
		python3 -m venv venv; \
		echo "Listo."; \
	fi

install-deps: requirements-runtime.txt install-venv
	@echo "Instalando dependencias de ejecución..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-runtime.txt
	@echo "Listo. (Para instalar todo requirements.txt: make deps-full)"

deps-full: install-venv
	@echo "Instalando requirements.txt completo..."
	$(PIP) install -r requirements.txt
	@echo "Listo."

run:
	@echo "Ejecutando main.py..."
	MPLBACKEND=Agg $(PYTHON) main.py

# Solo ej.2 (regresión con TXT), sin tocar tp_1/part_1.py — ver ej2_demo.py
ej2:
	MPLBACKEND=Agg $(PYTHON) ej2_demo.py

update-deps:
	@echo "Actualizando requirements.txt..."
	$(PIP) freeze > requirements.txt
	@echo "Hecho."
