# Resultados de Pruebas TP4 — Clasificador CNN

Todas las pruebas con 4 clases (Gatos, Aves, Caballos, Tortugas), 100 img/clase, validation 20%.

---

## 📊 TABLA COMPARATIVA

| # | IMG_SIZE | FILTERS | KERNEL | PADDING | DENSE | EPOCHS | **Acc** | **Loss** | Destacado |
|---|----------|---------|--------|---------|-------|--------|---------|----------|-----------|
| 1 | 512 | [32,64] | [7,7] | same | [256,128] | 30 | **47.5%** | 4.25 | ❌ |
| 2 | 64 | [32,64,128] | [3,3,3] | same | [256,128] | 30 | **56.25%** | 2.50 | |
| 3 | 64 | [32,64] | [5,3] | same | [256,128] | 30 | **57.5%** | 2.93 | |
| 4 | 64 | [32,64] | [3,3] | same | [64,32] | 30 | **56.25%** | 2.16 | Mejor loss |
| 5 | 128 | [32,64] | [3,3] | **valid** | [256,128] | 30 | **62.5%** | 2.22 | |
| 6 | 128 | [32,64] | [3,3] | valid | [256,128] | **50** | **51.25%** | 3.00 | Sobreajuste |
| 7 | 128 | [32,64] | [5,3] | valid | [256,128] | 50 | **52.5%** | 3.15 | |
| **8** 🏆 | 128 | **[32,64,128]** | [3,3,3] | valid | [256,128] | 50 | **67.5%** | 3.47 | **Mejor accuracy** |
| 9 | 128 | [32,64,128] | [5,3,3] | valid | [256,128] | **100** | **61.25%** | 3.66 | Sobreajuste |

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

### Prueba 3 — 57.5% ✅
```
IMG_SIZE=64, KERNEL=[5,3], FILTERS=[32,64]
```
> **Kernel 5×5 en primera capa** ve más contexto (25 píxeles vs 9 del 3×3).
> Las **Tortugas** suben a 0.70 de recall (antes 0.57).

### Prueba 4 — 56.25% ✅ (mejor loss)
```
IMG_SIZE=64, DENSE=[64,32], KERNEL=[3,3]
```
> **Loss más bajo: 2.16.** Menos neuronas densas = menos ruido.
> **Conclusión:** [64,32] es mejor que [256,128] para 64×64.

### Prueba 5 — 62.5% 🥈
```
IMG_SIZE=128, PADDING='valid', KERNEL=[3,3], EPOCHS=30
```
> **padding='valid'** reduce la imagen naturalmente (sin relleno de ceros).
> **Gatos 0.84 recall** — muy bueno.

### Prueba 6 — 51.25% ❌ (la peor después de la 1)
```
Misma config que prueba 5 pero con EPOCHS=50 (más épocas)
```
> **Empeoró** respecto a prueba 5 (62.5% → 51.25%). Esto es **SOBREAJUSTE**: más épocas no siempre es mejor, la red empieza a memorizar.

### Prueba 7 — 52.5% ⚠️
```
KERNEL=[5,3] con IMG_SIZE=128 y valid
```
> **Kernel 5×5 no ayuda** con IMG_SIZE=128 y valid. La imagen se reduce mucho más rápido.

### Prueba 8 — 67.5% 🏆 (LA MEJOR)
```
IMG_SIZE=128, FILTERS=[32,64,128], PADDING=valid, EPOCHS=50
```
> **3 capas Conv2D** aprenden patrones jerárquicos (bordes → texturas → formas).
> **Gatos 0.89 recall** — excelente.
> **Caballos 0.83 precision** — cuando predice caballo, acierta muy seguido.
> **Aves 0.56 recall** — siguen siendo lo más difícil.

### Prueba 9 — 61.25% ⚠️
```
Misma config que prueba 8 pero con KERNEL=[5,3,3] y EPOCHS=100
```
> **100 épocas es demasiado.** Aunque mejor que prueba 8 en theory, el accuracy bajó (67.5 → 61.25). 
> La red empieza a sobreajustar después de cierto punto.

---

## 📈 HALLAZGOS CLAVE

1. **padding='valid' es mejor que 'same'** — reduce progresivamente sin ceros artificiales
2. **3 capas Conv2D superan a 2 capas** — [32,64,128] dio el mejor resultado (67.5%)
3. **50 épocas es mejor que 100** — con pocos datos, más épocas = sobreajuste
4. **Gatos** son la clase más fácil (recall 0.74-0.89)
5. **Aves** son las más difíciles (recall 0.39-0.56) — plumas y texturas variadas
6. **Kernel 5×5 en primera capa** da resultados mixtos — ayuda con 64×64, no con 128×128+valid

---

## 🚀 PRÓXIMOS EXPERIMENTOS SUGERIDOS

| # | Configuración | Por qué |
|---|--------------|---------|
| **10** 🏆 | Prueba 8 + DENSE=[128,64] | Menos densas + 3 capas = menos sobreajuste |
| **11** 🔥 | Prueba 8 + DENSE=[64,32] + EPOCHS=80 | Red más chica, más épocas controladas |
| **12** 🧪 | IMG_SIZE=64, FILTERS=[32,64,128], valid, EPOCHS=50 | Misma config ganadora pero más rápido |
| **13** 📊 | MAX_POR_CLASE=200 (si hay) | Más datos siempre ayuda |