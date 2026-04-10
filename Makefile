#make
init: install-venv install-deps

PYTHON=venv\Scripts\python.exe
PIP=venv\Scripts\pip.exe

install-venv:
	@echo "Instalando entorno..."
	python -m venv venv
	@echo "Entorno creado."

install-deps: requirements.txt
	@echo "Instalando dependencias..."
	$(PIP) install -r requirements.txt
	@echo "Dependencias instaladas."

update-deps:
	@echo "Actualizando dependencias..."
	$(PIP) freeze > requirements.txt
	@echo "Dependencias actualizadas."