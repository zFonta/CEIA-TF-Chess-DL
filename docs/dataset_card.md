# Dataset card — `ceia-chess-eval`

Documentación del proceso de etiquetado, exigida por el **requerimiento 2.3**:
versión de Stockfish, profundidad de búsqueda y transformaciones aplicadas a los
datos.

- **Repositorio:** [`zFonta/ceia-chess-eval`](https://huggingface.co/datasets/zFonta/ceia-chess-eval) (Hugging Face Datasets)
- **Versión de configuración:** `v1` — [`configs/dataset_v1.yaml`](../configs/dataset_v1.yaml)
- **Licencia de los datos de origen:** CC0 (base pública de Lichess)

## La versión publicada, en números

Corrida completa ejecutada en Colab Pro (runtime de CPU, 2 vCPU asignadas) con
las notebooks `00`, `01` y `02`. Todos los valores de esta sección salen de la
salida guardada en esas notebooks, no de una estimación.

| Medida | Valor |
|---|---|
| Posiciones | **2.552.804** |
| Shards | 157 (`games_per_shard = 5000`) |
| Partidas distintas representadas | 772.797 |
| Posiciones por partida (efectivo) | 3,30 |
| Dump de origen | `2025-06`, único |
| Motor | **Stockfish 17.1**, profundidad 12 |
| Velocidad de etiquetado | ~12,1 posiciones/segundo con 2 workers |

**Composición por control de tiempo:**

| Categoría | Posiciones | Proporción |
|---|---|---|
| Blitz | 2.332.732 | 91,38 % |
| Rapid | 212.703 | 8,33 % |
| Classical | 7.369 | 0,29 % |

El dataset es, en los hechos, casi enteramente de Blitz. No es un desvío del
filtro sino su consecuencia: Blitz es con diferencia la categoría más jugada en
Lichess, y Classical es marginal incluso antes de aplicar el umbral de ELO.
Conviene tenerlo presente al interpretar los resultados del bloque 4.

**Distribución de las etiquetas y de las partidas:**

| Medida | Valor |
|---|---|
| Mueven blancas / mueven negras | 49,68 % / 50,32 % |
| `value_white` medio | +0,0463 (≈ **+18,5 centipeones**) |
| `value_white` desvío | 0,4880 |
| Posiciones con mate forzado | 36.486 (1,43 %) |
| ELO medio (blancas / negras) | 2.363 / 2.363 |
| Ply medio / mediano | 47,7 / 41 |
| Rango de ply | 0 – 598 |
| Posiciones en las primeras 10 medias jugadas | 1,95 % |

**Rendimiento del filtro sobre el dump:** se recorrieron 91.189.178 partidas y
pasaron el filtro 1.661.093, una tasa de aceptación del **1,82 %** (el survey
previo sobre 1.000.072 partidas había estimado 1,76 %, así que la estimación
resultó buena). De esas partidas aceptadas se consumió el **47 %** para llegar a
2,5 millones de posiciones: el extracto de un solo mes alcanza de sobra, y queda
margen para ampliar el dataset sin descargar otro dump.

**Deduplicación:** de las posiciones muestreadas se descartaron las repetidas
contra todo lo ya publicado. La tasa crece a medida que el dataset se llena
—1.560 duplicados en el primer shard, 2.558 en el quinto— porque cada shard
nuevo se compara contra un conjunto mayor. El rendimiento sobre el techo teórico
(4 posiciones × 5.000 partidas por shard) fue del 88,2 % en los primeros cinco
shards.

## Contenido

Pares *(posición de ajedrez, evaluación numérica)* para entrenamiento supervisado
de una red de evaluación de posiciones. Las posiciones provienen de partidas
públicas de Lichess entre jugadores fuertes; las evaluaciones las produce
Stockfish a profundidad controlada.

## Procedencia

| Aspecto | Valor |
|---|---|
| Fuente | [database.lichess.org](https://database.lichess.org) — dumps mensuales de partidas estándar |
| Meses incluidos | **`2025-06`**, único dump (se registra por fila en `src_dump`) |
| Licencia | CC0 — dominio público |
| Uso | Académico, Trabajo Final de la CEIA-FIUBA |

## Filtro aplicado a las partidas

| Criterio | Valor |
|---|---|
| ELO mínimo | **2200**, exigido a **ambos** jugadores |
| Controles de tiempo | Blitz, Rapid, Classical |
| Excluidos | UltraBullet, Bullet, Correspondence |
| Solo partidas rateadas | Sí |
| Largo mínimo | 20 medias jugadas |
| Terminaciones excluidas | Abandoned, Rules infraction, Unterminated |

**Por qué se excluyen Bullet y UltraBullet:** a esos ritmos la calidad del juego
cae fuerte incluso entre jugadores de 2200+, porque muchas jugadas las decide el
reloj y no la posición. Eso introduce ruido sistemático en las posiciones
muestreadas.

**Por qué se excluye Correspondence:** por el motivo opuesto — permite consultar
libros de aperturas y bases de finales, así que no refleja juego humano no
asistido.

## Muestreo de posiciones

| Criterio | Valor |
|---|---|
| Posiciones por partida | 4 |
| Estratificación | 2 con blancas al turno, 2 con negras (requerimiento 1.3) |
| Aperturas | **No se saltean**: se muestrea la partida completa |
| Posiciones terminales | Excluidas (jaque mate y ahogado no son evaluables) |
| Semilla | 20260909, derivada por partida con un digest estable |
| Deduplicación | Por los 4 primeros campos del FEN, global entre shards y sesiones |

## Etiquetado

| Aspecto | Valor |
|---|---|
| Motor | **Stockfish 17.1** — verificado: `sf_version` toma ese único valor en las 2.552.804 filas |
| Versión fijada por el script | `sf_17.1` (ver `scripts/setup_stockfish.sh`) |
| Profundidad de búsqueda | **12**, fija; registrada en `sf_depth` |
| Threads por motor | 1 (el paralelismo es por procesos, no por motor) |
| Hash por motor | 64 MB |

La versión no se escribe a mano: se lee del handshake UCI del propio motor en
tiempo de ejecución y se guarda en cada fila. Un dataset construido con otro
binario queda identificado como tal sin depender de que alguien lo recuerde.

## Transformaciones aplicadas

### Normalización de la evaluación (requerimientos 1.1 y 4.2)

```
value = tanh(cp / 400)              cp = 400 · atanh(value)
```

1. La evaluación de Stockfish se toma en centipeones desde el punto de vista de
   las blancas.
2. Se **recorta a ±2000**. Esto mantiene `|value| ≤ 0.9999`, con lo cual la
   transformación inversa nunca diverge — tampoco si la red satura y devuelve
   exactamente ±1.
3. Se aplica `tanh(cp / 400)`.

Un **mate en N** se lleva al límite del recorte (±2000 → ±0.9999). La información
que el recorte descarta se conserva en las columnas `is_mate` y `mate_in`.

### Perspectiva del jugador al turno

La entrada de la red se codifica siempre desde la perspectiva de quien mueve: si
mueven las negras, el tablero se espeja verticalmente y se intercambian los
colores. Por eso el dataset guarda la etiqueta en los dos puntos de vista:

```
value_white = value_stm   si mueven blancas
value_white = -value_stm  si mueven negras
```

`value_stm` es el **target de entrenamiento**; `value_white` es la escala de los
requerimientos 1.4 y 4.2, la que reporta el motor.

## Esquema

| Columna | Tipo | Descripción |
|---|---|---|
| `game_id` | string | Identificador de la partida en Lichess |
| `ply` | int16 | Media jugada dentro de la partida (0 = posición inicial) |
| `fen` | string | FEN completo de la posición |
| `pos_key` | uint64 | Clave de deduplicación (digest de los 4 primeros campos del FEN) |
| `turn_white` | bool | Verdadero si mueven las blancas |
| `cp_white` | int32 | Centipeones desde el punto de vista de las blancas, recortado |
| `cp_stm` | int32 | Centipeones desde el punto de vista de quien mueve |
| `value_white` | float32 | Evaluación normalizada, punto de vista de las blancas |
| `value_stm` | float32 | Evaluación normalizada, punto de vista de quien mueve — **target** |
| `is_mate` | bool | La evaluación es un mate forzado |
| `mate_in` | int16 | Distancia al mate, con signo desde el punto de vista de las blancas; 0 si no hay mate |
| `white_elo` | int16 | ELO del jugador de blancas |
| `black_elo` | int16 | ELO del jugador de negras |
| `result` | string | Resultado de la partida (`1-0`, `0-1`, `1/2-1/2`) |
| `time_control` | string | Categoría de Lichess (`Blitz`, `Rapid`, `Classical`) |
| `sf_depth` | int16 | Profundidad de búsqueda usada |
| `sf_version` | string | Versión de Stockfish que produjo la etiqueta |
| `src_dump` | string | Dump mensual de origen |

Formato: **Parquet** con compresión zstd, particionado en shards.

## Codificación a tensores

Los tensores **no** se almacenan: se calculan al vuelo desde el FEN con
`chessdl.encoding.board_to_tensor`. El etiquetado es la parte cara e irrepetible;
la codificación es barata y va a iterar durante el diseño de la red. Así se puede
cambiar la representación sin re-etiquetar nada.

Formato actual: **18 planos de 8×8**, `float32`.

| Planos | Contenido |
|---|---|
| 0–5 | Piezas propias: P, N, B, R, Q, K |
| 6–11 | Piezas del rival: P, N, B, R, Q, K |
| 12–13 | Enroque propio: corto, largo |
| 14–15 | Enroque del rival: corto, largo |
| 16 | Casilla de captura al paso |
| 17 | Reloj de 50 jugadas, escalado a [0, 1] |

El orden de los planos de pieza es arbitrario para la red — solo tiene que ser
idéntico en entrenamiento, validación e inferencia. Se eligió este porque
coincide con `chess.PIECE_TYPES` de python-chess, de modo que el índice sale
directo como `piece_type - 1`, sin tabla de traducción intermedia.

No hay plano de "turno": la orientación del tablero ya lo codifica.

## Validación de calidad

### Chequeos automáticos (requerimiento 3.2)

Corren sobre el dataset completo en cada build, vía
`python -m chessdl.scripts.validate_dataset`:

1. Sin posiciones duplicadas.
2. Balance de color: ninguna clase de turno supera el 55%.
3. Etiquetas dentro de `[-1, 1]`, sin nulos ni NaN.
4. Sin valores nulos en ninguna columna.
5. ELO de ambos jugadores por encima del umbral configurado.
6. Coherencia de signo entre `cp_white`, `cp_stm` y `turn_white`.
7. FENs parseables y no terminales.
8. Coherencia entre `is_mate`, `mate_in` y el límite de recorte.

Resultado sobre el dataset publicado (salida de la notebook `01`, sección final,
reproducida en la `02`): **los ocho chequeos pasan**. En particular, 2.552.804
claves distintas sobre 2.552.804 filas —cero duplicados en todo el dataset— y
ninguna columna con nulos.

La transformación inversa también se verificó sobre el dataset completo:
reconstruir los centipeones desde `value_white` y compararlos con la columna
`cp_white` da un error máximo de **0,0138 centipeones** y un error medio de
0,000009. Es el requerimiento 4.2 medido, no argumentado.

### Validación manual del etiquetado

Mitigación del Riesgo 4 del plan (errores sistemáticos en el etiquetado con
Stockfish). Evaluaciones sobre posiciones de referencia con valor conocido,
medidas con **Stockfish 17.1 a profundidad 12** —la misma versión y profundidad
que etiquetaron el dataset— a través de `chessdl.data.labeling.analyse_fen`, que
es el mismo camino de código que usa el pipeline:

| Posición | cp_white | value_white |
|---|---|---|
| Posición inicial | +33 | +0,0823 |
| Blancas con torre de más | +569 | +0,8901 |
| Final de peones equilibrado | −12 | −0,0300 |
| Rey y dama vs. rey, ventaja blancas (negras al turno) | +501 | +0,8490 |
| Rey y dama vs. rey, ventaja negras (blancas al turno) | −510 | −0,8551 |
| Mate en 1 para las blancas | +2000 (mate en 1) | +0,9999 |

Los valores son los esperados: la posición inicial queda apenas por encima de
cero, las ventajas materiales grandes dan valores altos pero no saturados, el
mate forzado sí satura en ±1, y las dos posiciones espejadas (filas 4 y 5, que
son la misma posición con los colores invertidos) dan el signo opuesto y
magnitudes que coinciden dentro del 2 %.

> Una corrección respecto de la versión anterior de esta tabla, que se había
> medido con Stockfish 16: las filas de rey y dama contra rey **no** saturan en
> ±2000. A profundidad 12 el motor todavía no ve el mate forzado desde esas
> posiciones y las evalúa por material, alrededor de ±500. Saturar requiere que
> el motor anuncie mate, que es lo que ocurre en la última fila.

### Prueba de consistencia sobre el dataset completo

Un control que no depende de posiciones elegidas a mano. Separando el dataset
según quién mueve y llevando las dos mitades a la escala de las blancas:

| Medida | Blancas al turno | Negras al turno (reflejada) |
|---|---|---|
| Media | +0,0897 | +0,0035 |
| Desvío | 0,4848 | 0,4869 |
| \|valor\| medio | 0,3829 | 0,3741 |

Los desvíos y las magnitudes medias coinciden: las dos mitades del dataset son
la misma distribución. La **diferencia de medias, +0,0862, no es un error**: es
el valor de tener la jugada. Traducido con la transformación inversa da
**+34,6 centipeones**, que coincide con la evaluación que el propio Stockfish le
da a la posición inicial en la tabla de arriba (+33 cp). Dos mediciones
independientes —una sobre 2,5 millones de posiciones reales, otra sobre una sola
posición— llegan al mismo número.

De ahí sale también la ventaja global de las blancas en el dataset, +0,0463
(+18,5 cp): es el promedio de las dos mitades, porque la mitad de las posiciones
tiene a las negras al turno.

## Limitaciones conocidas

- **La profundidad 12 es un compromiso.** Es suficiente para posiciones tácticas
  simples, pero no para posiciones donde el plan correcto está a más jugadas de
  distancia. Las etiquetas tienen ruido, y ese ruido es el techo de lo que la red
  puede aprender.
- **El umbral de ELO no garantiza calidad jugada a jugada.** Un jugador de 2200+
  igual comete errores; el filtro mejora la distribución de posiciones, no la
  corrección de cada movimiento.
- **Sesgo hacia posiciones desequilibradas.** Al muestrear uniformemente sobre
  toda la partida, los finales —donde la evaluación suele estar definida— aportan
  su parte, así que la distribución de etiquetas no está centrada en cero.
- **El dataset es casi todo Blitz (91,4 %).** El filtro admite Blitz, Rapid y
  Classical, pero la composición real de Lichess hace que Classical aporte el
  0,29 %. Si el bloque 4 mostrara que el ritmo de juego importa, la forma de
  corregirlo es muestrear por categoría, no cambiar el filtro.
- **Un solo mes de partidas (`2025-06`).** Alcanza para el volumen buscado, pero
  todo lo que sea estacional o propio de la meta de ese mes queda dentro del
  dataset sin contrapeso.
- **El etiquetado depende de que el binario de Stockfish sea sólido en la
  máquina.** Se detectó un entorno donde Stockfish 17.1 —tanto el build `avx2`
  como el portable `x86-64`— muere con SIGSEGV al buscar en finales con muy poco
  material, aunque analiza posiciones normales sin problema. En Colab, donde se
  construyó este dataset, no ocurre. El pipeline reintenta con un motor nuevo y
  descarta la posición que falla dos veces, así que el efecto sería perder
  finales en silencio; por eso `scripts/setup_stockfish.sh` ahora avisa cuando
  detecta ese comportamiento.
- **El recorte a ±2000 comprime el extremo superior.** Todas las ventajas
  decisivas se ven iguales para la red. Es deliberado: la diferencia entre "gana"
  y "gana más" no es información útil para elegir una jugada.

## Uso previsto y restricciones

Uso académico: entrenamiento de una red de evaluación de posiciones en el marco
del Trabajo Final de la CEIA-FIUBA.

**Este dataset y los modelos derivados no deben usarse para asistir a jugadores
en partidas competitivas en línea**: constituiría una violación de los términos
de servicio de las plataformas de ajedrez (requerimiento 5.2).
