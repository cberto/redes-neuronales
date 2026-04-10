#make
init: install-venv install-deps

PYTHON=venv/bin/python
PIP=venv/bin/pip

install-venv:
	@echo "Instalando entorno..."
	python3 -m venv venv
	@echo "Entorno creado."

install-deps: requirements.txt
	@echo "Instalando dependencias..."
	$(PIP) install -r requirements.txt
	@echo "Dependencias instaladas."

update-deps:
	@echo "Actualizando dependencias..."
	$(PIP) freeze > requirements.txt
	@echo "Dependencias actualizadas."