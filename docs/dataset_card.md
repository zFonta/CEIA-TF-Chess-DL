# Dataset card — `ceia-chess-eval`

Documentación del proceso de etiquetado, exigida por el **requerimiento 2.3**:
versión de Stockfish, profundidad de búsqueda y transformaciones aplicadas a los
datos.

- **Repositorio:** `zFonta/ceia-chess-eval` (Hugging Face Datasets)
- **Versión de configuración:** `v1` — [`configs/dataset_v1.yaml`](../configs/dataset_v1.yaml)
- **Licencia de los datos de origen:** CC0 (base pública de Lichess)

## Contenido

Pares *(posición de ajedrez, evaluación numérica)* para entrenamiento supervisado
de una red de evaluación de posiciones. Las posiciones provienen de partidas
públicas de Lichess entre jugadores fuertes; las evaluaciones las produce
Stockfish a profundidad controlada.

## Procedencia

| Aspecto | Valor |
|---|---|
| Fuente | [database.lichess.org](https://database.lichess.org) — dumps mensuales de partidas estándar |
| Meses incluidos | Ver la columna `src_dump` (se registra por fila) |
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
| Motor | Stockfish — la versión exacta se registra en la columna `sf_version` |
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

### Validación manual del etiquetado

Mitigación del Riesgo 4 del plan (errores sistemáticos en el etiquetado con
Stockfish). Evaluaciones sobre posiciones de referencia con valor conocido,
medidas a profundidad 12:

| Posición | cp_white | value_white |
|---|---|---|
| Posición inicial | +40 | +0.0997 |
| Blancas con torre de más | +543 | +0.8758 |
| Final de peones equilibrado | −1 | −0.0025 |
| Rey y dama vs. rey (ventaja blancas) | +2000 | +0.9999 |
| Rey y dama vs. rey (ventaja negras) | −2000 | −0.9999 |
| Mate en 1 para las blancas | +2000 (mate en 1) | +0.9999 |

Los valores son los esperados: la posición inicial queda cerca de cero con una
leve ventaja para las blancas, las ventajas materiales decisivas saturan en ±1, y
la misma posición espejada produce el signo opuesto.

> Los valores de esta tabla se midieron con Stockfish 16. Al regenerar el dataset
> con la versión fijada en `setup_stockfish.sh` conviene rehacer la tabla; las
> magnitudes deberían moverse poco y los signos nada.

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
- **El recorte a ±2000 comprime el extremo superior.** Todas las ventajas
  decisivas se ven iguales para la red. Es deliberado: la diferencia entre "gana"
  y "gana más" no es información útil para elegir una jugada.

## Uso previsto y restricciones

Uso académico: entrenamiento de una red de evaluación de posiciones en el marco
del Trabajo Final de la CEIA-FIUBA.

**Este dataset y los modelos derivados no deben usarse para asistir a jugadores
en partidas competitivas en línea**: constituiría una violación de los términos
de servicio de las plataformas de ajedrez (requerimiento 5.2).
