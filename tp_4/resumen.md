# Resultados de Pruebas TP4 — Clasificador CNN

Todas las pruebas con 4 clases (Gatos, Aves, Caballos, Tortugas), 100 img/clase, validation 20%.

---

## 📊 TABLA COMPARATIVA

| Prueba | IMG_SIZE | FILTERS | KERNEL | POOL_STRIDES | PADDING | DENSE | EPOCHS | **Acc** | **Loss** |
|--------|----------|---------|--------|-------------|---------|-------|--------|---------|----------|
| **1** ⚠️ | 512 | [32,64] | [7,7] | [2,2] | same | [256,128] | 30 | **47.5%** | 4.25 |
| **2** | 64 | [32,64,128] | [3,3,3] | [2,2,2] | same | [256,128] | 30 | **56.25%** | 2.50 |
| **3** ✅ | 64 | [32,64] | [5,3] | [2,2] | same | [256,128] | 30 | **57.5%** | 2.93 |
| **4** | 64 | [32,64] | [3,3] | [2,2] | same | [64,32] | 30 | **56.25%** | 2.16 |
| **5** 🏆 | 128 | [32,64] | [3,3] | [2,2] | **valid** | [256,128] | 30 | **62.5%** | 2.22 |

---

## 🔬 ANÁLISIS DE RESULTADOS

### Prueba 1 — 47.5% ❌ (peor)
```
IMG_SIZE=512, KERNEL=[7,7], PADDING=same
```
> **Problema:** 512×512 + kernel 7×7 + padding='same' mantiene el tamaño enorme → millones de parámetros → OOM o aprendizaje pobre.
> **Loss 4.25** es muy alto, señal de que la red no converge bien.

### Prueba 2 — 56.25% ✅ (subió 9 puntos)
```
IMG_SIZE=64, FILTERS=[32,64,128], KERNEL=[3,3,3], POOL_STRIDES=[2,2,2]
```
> **Acierto:** Bajar a 64×64 redujo drásticamente los parámetros.
> **3 capas Conv2D** ayudan a captar más patrones jerárquicos.
> **Loss baja de 4.25 → 2.50** (la red aprende mucho mejor).

### Prueba 3 — 57.5% ✅ (mejor hasta ese momento)
```
IMG_SIZE=64, KERNEL=[5,3], FILTERS=[32,64]
```
> **Kernel 5×5 en primera capa** ve más contexto (25 píxeles vs 9 del 3×3).
> Las **Tortugas** suben a 0.70 de recall (antes 0.57) → el kernel más grande ayuda a distinguir sus formas redondeadas.

### Prueba 4 — 56.25% ✅ (misma accuracy, mejor loss)
```
IMG_SIZE=64, DENSE=[64,32], KERNEL=[3,3]
```
> **Loss más bajo: 2.16** (contra 2.93 de prueba 3). Menos neuronas densas = menos ruido.
> **Tortugas 0.70** igual que prueba 3.
> **Conclusión:** [64,32] es mejor que [256,128] para 64×64 — menos sobreajuste.

### Prueba 5 — 62.5% 🏆 (la mejor)
```
IMG_SIZE=128, PADDING='valid', KERNEL=[3,3]
```
> **padding='valid'** reduce la imagen naturalmente (sin relleno de ceros).
> Con 128×128 + valid: 128→126→63→61→30 (aprox). Mucha más resolución que 64×64.
> **Gatos 0.84 recall** — muy bueno.
> **Aves 0.44 recall** — sigue siendo la clase más difícil (plumas, texturas finas).
> **Conclusión:** Más resolución + padding='valid' fue la mejor combinación.

---