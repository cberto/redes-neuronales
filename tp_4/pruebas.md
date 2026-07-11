## PRUEBA 1:

```
IMG_SIZE = 512
MAX_POR_CLASE = 100
KERNEL_SIZES = [7, 7]
POOL_STRIDES = [2, 2]
DENSE_UNITS = [256, 128]
EPOCHS = 30
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.52      0.74      0.61        19
        Aves       0.46      0.33      0.39        18
    Caballos       0.50      0.20      0.29        20
    Tortugas       0.44      0.61      0.51        23

    accuracy                           0.47        80

   macro avg       0.48      0.47      0.45        80
weighted avg       0.48      0.47      0.45        80

Accuracy (val) : 47.50%
Loss (val) : 4.2490

---

## PRUEBA 2:

```
FILTERS = [32, 64, 128]       # 3 capas en vez de 2
KERNEL_SIZES = [3, 3, 3]
POOL_STRIDES = [2, 2, 2]      # 64→32→16→8
IMG_SIZE = 64
EPOCHS = 30
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.61      0.74      0.67        19
        Aves       0.50      0.50      0.50        18
    Caballos       0.56      0.45      0.50        20
    Tortugas       0.57      0.57      0.57        23

    accuracy                           0.56        80

   macro avg       0.56      0.56      0.56        80
weighted avg       0.56      0.56      0.56        80

Accuracy (val) : 56.25%
Loss (val) : 2.4967

---

## PRUEBA 3:

```
KERNEL_SIZES = [5, 3]         # 1ra capa ve 5×5, 2da ve 3×3
IMG_SIZE = 64
FILTERS = [32, 64]
EPOCHS = 30
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.60      0.63      0.62        19
        Aves       0.50      0.44      0.47        18
    Caballos       0.53      0.50      0.51        20
    Tortugas       0.64      0.70      0.67        23

    accuracy                           0.57        80

   macro avg       0.57      0.57      0.57        80
weighted avg       0.57      0.57      0.57        80

Accuracy (val) : 57.50%
Loss (val) : 2.9262

---

## PRUEBA 4

```
DENSE_UNITS = [64, 32]        # Antes: [256, 128]
IMG_SIZE = 64
FILTERS = [32, 64]
KERNEL_SIZES = [3, 3]
EPOCHS = 30
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.57      0.68      0.62        19
        Aves       0.50      0.44      0.47        18
    Caballos       0.44      0.40      0.42        20
    Tortugas       0.70      0.70      0.70        23

    accuracy                           0.56        80

   macro avg       0.55      0.56      0.55        80
weighted avg       0.56      0.56      0.56        80

Accuracy (val) : 56.25%
Loss (val) : 2.1600

---

## PRUEBA 5

```
CONV_PADDING = "valid"
POOL_PADDING = "valid"
IMG_SIZE = 128                # más grande porque valid lo reduce
KERNEL_SIZES = [3, 3]
EPOCHS = 30
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.76      0.84      0.80        19
        Aves       0.53      0.44      0.48        18
    Caballos       0.69      0.45      0.55        20
    Tortugas       0.55      0.74      0.63        23

    accuracy                           0.62        80

   macro avg       0.63      0.62      0.61        80
weighted avg       0.63      0.62      0.62        80

Accuracy (val) : 62.50%
Loss (val) : 2.2153

---

## PRUEBA 6

```
CONV_PADDING = "valid"
POOL_PADDING = "valid"
IMG_SIZE = 128              
KERNEL_SIZES = [3, 3]
EPOCHS = 50 # mas epocas
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.44      0.63      0.52        19
        Aves       0.58      0.39      0.47        18
    Caballos       0.56      0.45      0.50        20
    Tortugas       0.52      0.57      0.54        23

    accuracy                           0.51        80
   macro avg       0.53      0.51      0.51        80
weighted avg       0.53      0.51      0.51        80

Accuracy (val) : 51.25%
Loss (val)     : 2.9980

---

## PRUEBA 7

```
CONV_PADDING = "valid"
POOL_PADDING = "valid"
IMG_SIZE = 128 
KERNEL_SIZES = [5, 3] #aumento el filtro del primero conv
EPOCHS = 50
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.58      0.74      0.65        19
        Aves       0.56      0.50      0.53        18
    Caballos       0.58      0.35      0.44        20
    Tortugas       0.43      0.52      0.47        23

    accuracy                           0.53        80
   macro avg       0.54      0.53      0.52        80
weighted avg       0.53      0.53      0.52        80

Accuracy (val) : 52.50%
Loss (val)     : 3.1507

---

## PRUEBA 8

```
CONV_PADDING = "valid"
POOL_PADDING = "valid"
IMG_SIZE = 128 
FILTERS = [32, 64, 128]
EPOCHS = 50
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.65      0.89      0.76        19
        Aves       0.56      0.56      0.56        18
    Caballos       0.83      0.50      0.62        20
    Tortugas       0.71      0.74      0.72        23

    accuracy                           0.68        80
   macro avg       0.69      0.67      0.66        80
weighted avg       0.69      0.68      0.67        80

Accuracy (val) : 67.50%
Loss (val)     : 3.4651

---

## PRUEBA 9

```
CONFIGURACION 8
KERNEL = [5, 3, 3]
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.65      0.89      0.72        19
        Aves       0.55      0.56      0.54        18
    Caballos       0.80      0.50      0.58        20
    Tortugas       0.70      0.74      0.70        23

    accuracy                           0.65        80
   macro avg       0.67      0.63      0.62        80
weighted avg       0.67      0.63      0.62        80

Accuracy (val) : 61.25%
Loss (val)     : 3.6635

---


## PRUEBA 10 - (se agregan todas las clases que subio la cursada)

```
CONFIGURACION 8
SE AGREGAN LAS 9 CLASES
```

Reporte de Clasificación:
              precision    recall  f1-score   support

       Gatos       0.33      0.29      0.31        24
        Aves       0.22      0.29      0.25        17
    Caballos       0.33      0.40      0.36        20
    Tortugas       0.47      0.44      0.46        18
     Conejos       0.46      0.59      0.52        22
 Hipopótamos       0.40      0.27      0.32        22
      Perros       0.15      0.25      0.19        16
   Pingüinos       0.11      0.04      0.06        25
  Serpientes       0.56      0.56      0.56        16

    accuracy                           0.34       180
   macro avg       0.34      0.35      0.34       180
weighted avg       0.33      0.34      0.33       180

  Accuracy (val) : 33.89%
  Loss (val)     : 9.5001
---