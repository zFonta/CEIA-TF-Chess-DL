# Alcance del bloque 4 — Red neuronal

Documento de alcance de la tarea 4 del WBS (*Desarrollo de la red neuronal*,
144 h). Fija qué se construye, qué decisiones ya están tomadas y por qué, y qué
queda explícitamente afuera. Se escribe **antes** de codificar para que la
comparación entre arquitecturas sea una comparación y no una acumulación de
diferencias.

## Qué se entrega

| Entregable del plan | Cómo se cumple |
|---|---|
| Pesos de la red en formato estándar (`.pt`) | Repositorio de modelos en Hugging Face |
| Registro del experimento (hiperparámetros, curvas, métricas) | Mismo repositorio, junto a cada checkpoint |
| Diagrama de la arquitectura de la red | `docs/` — se genera al cerrar 4.2 y 4.8 |
| Métricas sobre validación y test | Informe de 4.7 |

## Las dos arquitecturas

El plan pone la ResNet como **línea base obligatoria** (tarea 4.2) y el
transformer como **exploración opcional** (tarea 4.8, requerimiento 6.1,
historia 3). Por decisión del autor, en este proyecto el transformer se trata
como obligatorio: la comparación entre ambas es el aporte central del trabajo.

Eso no cambia el orden. La ResNet se entrena, se ajusta y se evalúa completa
(4.2 → 4.7) antes de empezar el transformer. Si el tiempo se corta, lo que queda
entregado es una línea base sólida y no dos modelos a medio hacer.

### Por qué se implementan desde cero

**No se usan arquitecturas pre-entrenadas ni de librería.** No es purismo
académico, es que no aplican:

- La entrada es un tensor de **(18, 8, 8)**. Una ResNet de `torchvision` empieza
  con una convolución 7×7 de stride 2 seguida de max-pooling: en dos pasos deja
  el tablero en 2×2. Está diseñada para imágenes de 224×224, no para un tablero.
- Los pesos pre-entrenados de ImageNet o de un ViT no transfieren nada a una
  representación de ajedrez: no hay texturas ni bordes que reutilizar.
- La ResNet de ajedrez es corta —convolución 3×3 con padding 1, que mantiene el
  tablero en 8×8 en toda la red, más bloques residuales y una cabeza de valor—.
  Implementarla es más simple que doblegar una librería, y la tarea 4.2 pide
  exactamente eso.
- Para el transformer sí se usan los bloques de `torch.nn`
  (`MultiheadAttention`, `TransformerEncoderLayer`): reimplementar la atención a
  mano no aporta nada y agrega superficie de error. Lo que se escribe es la
  arquitectura —tokenización, embeddings posicionales, cabeza—, que es la parte
  que hay que justificar y diagramar.

### Tokenización del transformer

**Una casilla = un token**, 64 tokens de longitud fija. Los embeddings
posicionales son **aprendidos** (una tabla de 64 × `d_model`): el tablero tiene
geometría fija y conocida, no hace falta nada sinusoidal.

Hay un detalle de la codificación que conviene tener presente: de los 18 planos,
**12 son espaciales** (las piezas) y **6 son constantes sobre todo el tablero**
(los cuatro de enroque, el reloj de 50 jugadas; el de captura al paso es
espacial). Para una CNN, difundir un escalar sobre los 64 casilleros es lo
normal. Para un transformer es redundante: repite el mismo valor en los 64
tokens.

Se arranca igual con los 18 planos en ambas arquitecturas, **para que la entrada
sea idéntica y la diferencia observada sea atribuible a la arquitectura**. La
variante de "64 tokens de pieza + 1 token de estado" queda como experimento de
la tarea 4.5, no como punto de partida.

### Presupuesto de parámetros

Comparar arquitecturas con tamaños distintos no responde la pregunta: si el
transformer gana con el triple de parámetros, no se sabe si ganó la atención o
el tamaño. Se arranca con presupuestos equiparados:

| Arquitectura | Configuración inicial | Parámetros |
|---|---|---|
| ResNet | 128 canales × 8 bloques residuales | ~2,9 M |
| Transformer | `d_model` 192, 6 capas, 8 cabezas | ~2,7 M |

Ambas se entrenan además con el mismo presupuesto de épocas y el mismo
optimizador antes de cualquier ajuste específico.

## Decisiones ya tomadas

### Target y escala

Se entrena contra **`value_stm`**, la etiqueta en perspectiva del jugador al
turno, que es la que corresponde a la entrada espejada. `value_white` se obtiene
con el cambio de signo y es lo que reportan el motor y la CLI (requerimientos 1.4
y 4.2).

### Partición: por partida, no por posición

**Es la decisión metodológica más importante del bloque.** El dataset tiene 4
posiciones por partida sobre 772.797 partidas. Una partición aleatoria *por
posición* deja posiciones de la misma partida en entrenamiento y en test: comparten
apertura, jugadores y estructura, y la métrica de validación queda inflada.

La deduplicación global por `pos_key` elimina los duplicados exactos, pero no
esta correlación. La partición se hace **por `game_id`**, con semilla fija:

```
entrenamiento 90 %   ·   validación 5 %   ·   test 5 %
```

La columna ya está en el dataset, así que hacerlo bien no cuesta nada. Hacerlo
mal no se detecta hasta que el motor juega contra Stockfish y rinde peor de lo
que decían las métricas.

### Función de pérdida (tarea 4.3)

Punto de partida **MSE** sobre `value_stm`. La alternativa a evaluar en 4.5 es
**Huber / smooth L1**, y hay un motivo concreto para probarla: el 1,43 % de las
posiciones son mates que saturan en ±0,9999, y el recorte a ±2000 centipeones
apila masa en los extremos. MSE pondera esos casos por el cuadrado del error.

### Piso de referencia

Antes de cualquier red se miden dos baselines no neuronales, que cuestan una hora
y hacen interpretable el RMSE:

1. **Predecir la media constante.** El piso ya se conoce del análisis
   exploratorio: RMSE = **0,488** (el desvío de `value_white`). Cualquier modelo
   tiene que ganarle; si no le gana, hay un error de implementación, no un
   problema de arquitectura.
2. **Conteo de material** (regresión lineal sobre los planos de pieza). Es el
   piso "sabe algo de ajedrez" contra el que se mide cuánto aporta la red.

## Persistencia y reanudación en Hugging Face

Mismo criterio que el pipeline de datos: **nada durable en disco local**. Se suma
un tercer repositorio, esta vez de tipo *Model*:

| Repositorio | Contenido |
|---|---|
| `zFonta/ceia-chess-eval` | Dataset (ya publicado) |
| `zFonta/ceia-chess-work` | Extracto PGN y estado del pipeline de datos |
| `zFonta/ceia-chess-models` | **Nuevo:** checkpoints, configuración y métricas |

Cada checkpoint guarda pesos, estado del optimizador y del scheduler, época,
estado de los generadores aleatorios y el historial de métricas. Con eso una
desconexión de Colab cuesta como mucho una época, igual que en el pipeline de
datos una desconexión costaba como mucho un shard.

Se conservan el **último** checkpoint y el **mejor por métrica de validación**;
los intermedios se van pisando para no llenar el repositorio. El historial de
métricas se sube como archivo aparte en cada época, así las curvas de pérdida
—que son un entregable del plan— sobreviven aunque se pierda el runtime.

## Métricas (tarea 4.7)

| Métrica | Para qué |
|---|---|
| RMSE sobre `value` | Métrica primaria de entrenamiento y comparación |
| MAE en centipeones | Escala interpretable, vía transformación inversa |
| Acuerdo de signo | ¿Acierta quién está mejor? Es lo que usa la búsqueda |
| Correlación de Spearman | Ordenamiento, que es lo que importa para elegir jugada |

Una advertencia sobre el RMSE en centipeones: la transformación inversa es **no
lineal** y diverge cerca de ±1, así que un RMSE calculado en centipeones queda
dominado por las posiciones saturadas y dice más sobre los mates que sobre el
juego normal. Por eso la métrica primaria se reporta en escala de `value`, y en
centipeones se reporta MAE, que es menos sensible a esa cola.

## Consideraciones y riesgos

### Sobre el tiempo de respuesta

El requerimiento 1.7 (5 segundos por jugada, búsqueda incluida) **no se verifica
en este bloque**. Depende de dónde corra el motor, y el motor todavía no existe:
su verificación es del bloque 5.

Lo único que se hace acá es **registrar el tiempo de inferencia por lote** en el
entorno donde se entrena, como dato de referencia. Es gratis de medir y le ahorra
trabajo al bloque 5, pero no condiciona la arquitectura.

### Entorno de entrenamiento: GPU T4, runtime estándar

El entrenamiento corre en **Colab Pro** (plan de USD 10, ~100 unidades de cómputo
mensuales). La recomendación es el **runtime de GPU estándar (T4), sin high-RAM**,
y conviene justificarla porque el reflejo natural es pedir la GPU más grande
disponible.

El modelo es chico y el tablero es de 8×8. Estimando el costo por época sobre el
90 % de entrenamiento:

| Arquitectura | FLOPs/muestra | Por época | T4 | L4 | A100 |
|---|---|---|---|---|---|
| ResNet 128×8 | 306 M | 2,1 PFLOPs | ~2,3 min | ~0,9 min | ~0,4 min |
| Transformer d=192 ×6 | 359 M | 2,5 PFLOPs | ~2,7 min | ~1,0 min | ~0,4 min |

Una campaña de 50 épocas en T4 son unas **2 horas**. Pasar a A100 la bajaría a
~20 minutos, pero **quema unidades del orden de seis veces más rápido**: con 100
unidades mensuales, la T4 rinde decenas de horas y la A100 se agota en una tarde.
Para un modelo de 3 M de parámetros sobre entradas de 8×8 ese gasto no compra
nada, porque la GPU grande queda desaprovechada: con dimensiones espaciales tan
chicas, la utilización es baja sin importar el hardware.

Las estimaciones de la tabla tienen margen de error amplio —la utilización real
sobre tensores de 8×8 es difícil de predecir— y se reemplazan por la medición
efectiva en la tarea 4.1. El orden de magnitud sí es confiable, y es el que
sustenta la decisión.

Tres detalles del runtime que importan más que el modelo de GPU:

- **Lotes grandes** (1.024 a 4.096). Con 8×8 de resolución, un lote chico deja la
  GPU haciendo nada entre kernels. Es la palanca más efectiva de las tres.
- **Precisión mixta (AMP)**. La T4 tiene tensor cores; usar `float16` en el
  forward es prácticamente gratis en código y cambia bastante el tiempo.
- **RAM estándar alcanza.** El caché de tensores son 2,9 GB en `uint8` contra los
  12,7 GB del runtime estándar. High-RAM gastaría más unidades sin necesidad.

No se usa **TPU**: PyTorch sobre XLA agrega complejidad de compilación y de
depuración que no se justifica para un modelo de este tamaño.

Si durante el ajuste de hiperparámetros (4.5) hiciera falta iterar más rápido, el
escalón sensato es **L4**, no A100.

### El runtime de Colab cambia de signo

Toda la fase de datos pedía **CPU** y `colab.py` avisa si detecta una GPU. Para
entrenar eso se invierte. La función `describe_runtime()` necesita saber en qué
fase está, o va a dar un consejo equivocado con mucha seguridad.

### Composición del dataset

El dataset es **91,4 % Blitz** (ver `dataset_card.md`). No invalida nada, pero al
reportar resultados conviene desglosar por `time_control`: es una línea de código
y responde de antemano una pregunta previsible del director.

### El cuello de botella esperado es el dataloader, no la GPU

Codificar una posición por el camino directo (FEN → `Board` → tensor) cuesta
**79 µs**, de los cuales 44 µs son parsear el FEN. Son ~12.600 posiciones por
segundo y por núcleo; con las 2 vCPU que Colab viene asignando, unos 100 segundos
por época solo en codificar, con la GPU esperando.

Puesto al lado del costo de GPU de la sección anterior, el problema se ve claro:
la T4 necesita ~140 segundos por época y codificar sobre la marcha cuesta ~100.
Son del mismo orden, así que sin caché el tiempo por época casi se duplica y la
mitad del gasto de unidades se va en esperar a la CPU.

La salida es precomputar los tensores una sola vez: **2,9 GB en `uint8`** (los
planos son binarios salvo el reloj de 50 jugadas), que entra en memoria o se mapea
de disco, y se castea a `float32` en GPU. El costo es de unos dos minutos de CPU
una única vez, y a partir de ahí el dataloader es rebanar un array.

Esto **no modifica el dataset**: el Parquet sigue guardando FENs, que fue la
decisión de diseño para poder cambiar la representación sin re-etiquetar. Es un
caché de entrenamiento, descartable.

## Fuera de alcance

- **Todo lo que no sea predecir la evaluación.** La red tiene una única tarea:
  dado un FEN, replicar el puntaje que le asigna Stockfish. No hay cabeza de
  política ni predicción de jugadas. La historia 2 menciona "predice
  movimientos", pero el requerimiento 1.6 define la selección de jugada como
  **búsqueda de un nivel sobre la función de evaluación**: el motor elige
  evaluando las posiciones resultantes, no clasificando jugadas. Si el bloque 6
  necesita una métrica de acuerdo de jugada contra Stockfish, se calcula ahí
  —basta correr el motor y el de referencia sobre las mismas posiciones—, no
  requiere nada guardado en el dataset.
- **Entrenamiento por refuerzo / self-play.** Excluido explícitamente en el
  alcance del plan.
- **Arquitectura híbrida ResNet + Transformer.** Es una combinación razonable
  —es por donde fueron los motores modernos— pero responde una pregunta distinta
  de la que plantea este trabajo, y triplicaría la superficie de ajuste de
  hiperparámetros cuando la tarea 4.5 asigna 30 h para una arquitectura. Queda
  como **trabajo futuro** de la memoria, respaldado por los resultados de la
  comparación en vez de como especulación.
- **El motor de juego y su evaluación** (bloques 5 y 6 del WBS).

## Secuencia de trabajo

| Tarea del plan | Contenido |
|---|---|
| 4.1 Entorno de entrenamiento (12 h) | `Dataset` de PyTorch, partición por partida, caché de tensores, baselines de referencia, notebook `03` |
| 4.2 Arquitectura residual (30 h) | ResNet + diagrama de arquitectura |
| 4.3 Función de pérdida (16 h) | MSE, con Huber como alternativa instrumentada |
| 4.4 Primera campaña (20 h) | Entrenamiento y diagnóstico contra los baselines |
| 4.5 Hiperparámetros (30 h) | Ajuste, y las variantes marcadas arriba como experimentos |
| 4.6 Segunda campaña (20 h) | Configuración optimizada |
| 4.7 Validación y test (16 h) | Métricas sobre el split de test, tiempo de inferencia por lote, desglose por control de tiempo |
| 4.8 Transformer (36 h) | Arquitectura, entrenamiento con el mismo presupuesto, comparación |
