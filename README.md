# TPs-AA

## Aprendizaje Automático

### Trabajo Práctico 1

Para la realización de los ejercicios, se consideran los datos del archivo **Préstamo.csv** los cuales fueron descargados del enlace:  
[https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data).

Este conjunto de datos contiene información sobre personas que solicitaron un crédito bancario.

---

## Ejercicio 1

Se desea conocer el estado de la solicitud del préstamo (**OTORGADO** o **RECHAZADO**) para personas de 50 años conociendo su sexo, su máximo nivel educativo alcanzado, su estado de vivienda (si posee una propiedad propia, posee una hipoteca o alquila) y sabiendo si posee préstamos previos impagos.

1. Considerar los primeros ejemplos que cubren el **75%** del total de ejemplos como conjunto de entrenamiento y el resto como conjunto de prueba.
2. Implementar el algoritmo **FIND-S**.
3. Aplicar la hipótesis obtenida para predecir el estado de la solicitud del préstamo en el conjunto de prueba.

---

## Ejercicio 2

A partir de los resultados obtenidos en el **Ejercicio 1**, se pide lo siguiente:

1. Calcular la **matriz de confusión**.
2. Calcular el **accuracy**, el **recall**, la **especificidad** y la **precisión**.
3. Calcular el **F1-score**.
4. Calcular la **tasa de verdaderos positivos** y la **tasa de falsos positivos**.

---

## Ejercicio 3

Se desea conocer el estado de la solicitud del préstamo (**OTORGADO** o **RECHAZADO**) para personas de entre 40 y 45 años a partir de los atributos categóricos.

1. Dividir aleatoriamente el conjunto de datos para utilizar una parte de los mismos como conjunto de entrenamiento (**80%**) y otro como conjunto de prueba (**20%**).
2. Utilizar el clasificador **Naïve Bayes** para clasificar los ejemplos del conjunto de prueba.
3. Construir la **matriz de confusión** para el conjunto de prueba.
4. Calcular el **accuracy** y el **F1-score**.
5. Graficar la **curva ROC** para el conjunto de prueba.

---

**Fuente del dataset:**  
[https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
