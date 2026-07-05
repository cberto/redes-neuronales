# PRUEBA 1:
  1 capa oculta: relu (17)
  salida: sigmoid
  latente: relu
  optimizer: adam
  loss: binary_crossentropy
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step - loss: 0.1499 - val_loss: 2.6223

 Modelos exportados exitosamente en: /mnt/e/curso/Redes-Neuronales/TPs-RN/tp_3/models
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step

========================================
VERIFICACIÓN DE RECONSTRUCCIÓN
========================================

Carácter: 'space'
ORIGINAL:           RECONSTRUCCIÓN:
                 
                 
                 
                 
                 
                 
                 

Carácter: '!'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
  █           █  
  █           █  
  █           █  
  █           █  
                 
  █          ██  

Carácter: ''''
ORIGINAL:           RECONSTRUCCIÓN:
 █  █        █  █
 █  █        █  █
█  █        █  █ 
                 
                 
                 
                 

Carácter: '#'
ORIGINAL:           RECONSTRUCCIÓN:
 █ █         █ █ 
 █ █         █ █ 
█████       █████
 █ █         █ █ 
█████       █████
 █ █         █ █ 
 █ █         █ █ 

Carácter: '$'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
 ████        ████
█ █          ███ 
 ███         ███ 
  █ █         █ █
████        █ ██ 
  █           █  
Resultado 1_3:  {}
# PRUEBA 2:
  1 capa oculta: relu (17)
  salida: sigmoid
  latente: linear
  optimizer: adam
  loss: binary_crossentropy
Epoch 2000/2000
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - loss: 0.0589 - val_loss: 1.9343

 Modelos exportados exitosamente en: /mnt/e/curso/Redes-Neuronales/TPs-RN/tp_3/models
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step

========================================
VERIFICACIÓN DE RECONSTRUCCIÓN
========================================

Carácter: 'space'
ORIGINAL:           RECONSTRUCCIÓN:





              █


Carácter: '!'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
  █           █  
  █           █  
  █           █  
  █           █  
                 
  █           █  

Carácter: ''''
ORIGINAL:           RECONSTRUCCIÓN:
 █  █        █  █
 █  █        █  █
█  █        █  █ 
                 
                 
                 
                 

Carácter: '#'
ORIGINAL:           RECONSTRUCCIÓN:
 █ █         █ █ 
 █ █         █ █
█████       █████
 █ █         █ █
█████       █████
 █ █         █ █
 █ █         █ █ 

Carácter: '$'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
 ████        █ ██
█ █         █ █  
 ███         ███ 
  █ █         █ █
████        ████
  █           █  

# PRUEBA 3:
  1 capa oculta: relu (17)
  salida: tanh
  latente: linear
  optimizer: adam
  loss: mse
Epoch 2000/2000
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.0733 - val_loss: 0.4884

 Modelos exportados exitosamente en: /mnt/e/curso/Redes-Neuronales/TPs-RN/tp_3/models
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step

========================================
VERIFICACIÓN DE RECONSTRUCCIÓN
========================================

Carácter: 'space'
ORIGINAL:           RECONSTRUCCIÓN:
                 
                 
                 
                 
                 
                 
                 

Carácter: '!'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
  █           █  
  █           █  
  █           █  
  █           █  
                 
  █           █  

Carácter: ''''
ORIGINAL:           RECONSTRUCCIÓN:
 █  █        █  █
 █  █           █
█  █           █ 
                 
                 
                 
                 

Carácter: '#'
ORIGINAL:           RECONSTRUCCIÓN:
 █ █         █ █ 
 █ █         █ █ 
█████       █████
 █ █         █ █ 
█████       █████
 █ █         █ █ 
 █ █         █ █ 

Carácter: '$'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
 ████        ████
█ █         █ █  
 ███         ███ 
  █ █         █ █
████        █ ██ 
  █           █  

# PRUEBA 4:
  2 capa oculta: relu (24, 12)
  salida: tanh
  latente: linear
  optimizer: adam
  loss: mse
Epoch 2000/2000
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0731 - val_loss: 0.9571

 Modelos exportados exitosamente en: /mnt/e/curso/Redes-Neuronales/TPs-RN/tp_3/models
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step

========================================
VERIFICACIÓN DE RECONSTRUCCIÓN
========================================

Carácter: 'space'
ORIGINAL:           RECONSTRUCCIÓN:
                 
                 
                 
                 
                 
                 
                 

Carácter: '!'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
  █           █  
  █           █  
  █           █  
  █           █  
              █  
  █           █  

Carácter: ''''
ORIGINAL:           RECONSTRUCCIÓN:
 █  █        █  █
 █  █        █  █
█  █        █  █ 
                 
                 
                 
                 

Carácter: '#'
ORIGINAL:           RECONSTRUCCIÓN:
 █ █         █ █ 
 █ █         █ █ 
█████       █████
 █ █         █ █ 
█████       █████
 █ █         █ █ 
 █ █         █ █ 

Carácter: '$'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
 ████        ████
█ █         █ █  
 ███         ███ 
  █ █        ██ █
████        ████ 
  █           █  

# PRUEBA 5:
  2 capa oculta: relu (24, 12)
  salida: tanh
  latente: linear
  optimizer: adam
  loss: mse
  DENOISING: on

Epoch 2000/2000
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1427 - val_loss: 0.9485

 Modelos exportados exitosamente en: /mnt/e/curso/Redes-Neuronales/TPs-RN/tp_3/models
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step

========================================
VERIFICACIÓN DE RECONSTRUCCIÓN
========================================

Carácter: 'space'
ORIGINAL:           RECONSTRUCCIÓN:
                 
                 
                 
                 
                 
                 
                 

Carácter: '!'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
  █           █  
  █           █  
  █           █  
  █           █  
              █  
  █           █  

Carácter: ''''
ORIGINAL:           RECONSTRUCCIÓN:
 █  █        █  █
 █  █           █
█  █        █  █ 
                 
                 
                 
                 

Carácter: '#'
ORIGINAL:           RECONSTRUCCIÓN:
 █ █         █ █ 
 █ █         █ █ 
█████       ██ ██
 █ █         █ █ 
█████       █████
 █ █           █ 
 █ █         █ █ 

Carácter: '$'
ORIGINAL:           RECONSTRUCCIÓN:
  █           █  
 ████        ████
█ █         █ █  
 ███         ███ 
  █ █         █ █
████        ████ 
  █           █  