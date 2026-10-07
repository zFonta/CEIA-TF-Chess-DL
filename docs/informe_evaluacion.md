# Informe de evaluación

Resultados del sistema en un solo lugar: qué tan bien la red reproduce la
evaluación de Stockfish (RMSE) y qué tan bien juega el motor (pérdida en
centipeones y partidas contra Stockfish). Es el entregable *Informe de
evaluación* del plan.

Cada número sale de una notebook versionada con la salida de su corrida, que se
cita al lado. Las corridas son en Google Colab con una GPU T4.

| fuente | qué mide |
|---|---|
| [`05_hyperparameters`](../notebooks/05_hyperparameters.ipynb), [`07_transformer_tuning`](../notebooks/07_transformer_tuning.ipynb) | RMSE de test de las campañas finales |
| [`08_motor_y_partidas`](../notebooks/08_motor_y_partidas.ipynb) | Desglose del RMSE, tiempo por jugada, pérdida en centipeones, posiciones de referencia y partidas |
| [`apendice/lichess_3_elo`](../notebooks/apendice/lichess_3_elo.ipynb) | Las mismas mediciones sobre los modelos entrenados con más datos |

## 1. Los modelos evaluados

| | arquitectura | parámetros | corrida en el Hub |
|---|---|---|---|
| **ResNet** | 8 bloques residuales de 128 canales | 2.913.345 | `campana2-warmup` |
| **Transformer** | encoder de 6 capas sobre 64 casillas | 2.735.361 | `transformer-campana2-lotes-chicos` |

Los dos se entrenaron con el mismo dataset —2,55 millones de posiciones
etiquetadas con Stockfish 17.1 a profundidad 12 (ver
[`dataset_card.md`](dataset_card.md))— y se evalúan sobre el mismo split de test
de 127.499 posiciones, con partición por partida: ninguna posición de test
comparte partida con una de entrenamiento.

## 2. Precisión de la evaluación

### RMSE sobre el test

El target es la evaluación de Stockfish normalizada a [−1, 1] con
`tanh(cp / 400)`, desde el jugador al turno.

| | RMSE | R² | MAE (cp) | acuerdo de signo | Spearman |
|---|---|---|---|---|---|
| Piso: media constante | 0,4886 | 0,000 | — | — | — |
| Piso: material lineal | 0,3973 | 0,339 | — | — | — |
| **ResNet** | **0,2511** | **0,736** | **105,8** | **87,80 %** | **0,8385** |
| **Transformer** | **0,2531** | **0,732** | **109,0** | **86,80 %** | **0,8315** |

Las dos arquitecturas **empatan**: 0,80 % de diferencia, menos de lo que se
esperaría entre dos semillas de la misma configuración. Las dos mejoran el piso
de material un 36 %.

### Desglose (notebook 08, sección 5)

**Por control de tiempo.** El dataset es 91 % Blitz; las otras categorías no
empeoran el error.

| | Blitz (116.471) | Rapid (10.603) | Classical (425) |
|---|---|---|---|
| ResNet | 0,2506 | 0,2576 | 0,2255 |
| Transformer | 0,2529 | 0,2560 | 0,2288 |

**Acuerdo de signo según qué tan pareja está la posición.** Es lo que predice
cómo elige el motor, porque la búsqueda decide entre posiciones de valor
parecido.

| franja (\|valor\|) | posiciones | ResNet | Transformer |
|---|---|---|---|
| casi igualada (< 0,05) | 19.340 | 60,2 % | 61,3 % |
| ligera ventaja (0,05–0,20) | 33.256 | 77,6 % | 75,8 % |
| ventaja clara (0,20–0,50) | 29.833 | 88,1 % | 87,4 % |
| decidida (> 0,50) | 45.070 | 95,1 % | 94,5 % |

En las posiciones casi igualadas la red acierta quién está mejor apenas un poco
más que una moneda: es la mayor limitación de la evaluación.

**Requerimiento 1.4.** El 100 % de la salida de los dos modelos sobre las
127.499 posiciones de test queda dentro de [−1, 1].

### Posiciones de referencia (notebook 08, sección 7.2)

Doce posiciones de valor conocido, de lo parejo a lo decidido, evaluadas por la
red sin búsqueda y comparadas con Stockfish a profundidad 12 (requerimientos 1.4
y 4.2). Las dos redes dan el signo de Stockfish en **9 de 9** posiciones con
ventaja clara, y quedan a menos de 100 cp de cero en las parejas (2 de 3 la
ResNet, 3 de 3 el Transformer). En la dama colgada —una táctica de una jugada—
la ResNet da −324 cp donde Stockfish da −729: la ve, pero a medias.

## 3. Calidad de juego

### Tiempo por jugada (requerimiento 1.7, notebook 08 sección 6)

Percentil 99 sobre 100 posiciones de test con entre 1 y 59 jugadas legales:

| | 1 ply | 2 plies | 3 plies |
|---|---|---|---|
| ResNet | 23 ms | 388 ms | 8.588 ms |
| Transformer | 13 ms | 583 ms | 14.523 ms |

Uno y dos plies entran con holgura en los 5 segundos del requerimiento (9 a 372
veces de margen). Tres no entra, y por eso no se juega.

### Pérdida media en centipeones (notebook 08, sección 7)

Para cada una de 400 posiciones de test, cuánto pierde la jugada elegida frente
a la mejor según Stockfish a profundidad 12:

| motor | ACPL | mediana | acuerdo con SF | errores > 300 cp |
|---|---|---|---|---|
| ResNet, 1 ply | 176,8 ± 14,4 | 52 | 30,5 % | 21,0 % |
| Transformer, 1 ply | 257,1 ± 18,6 | 84 | 28,5 % | 34,2 % |
| **ResNet, 2 plies** | **87,4 ± 11,2** | **25** | **39,0 %** | **5,5 %** |
| **Transformer, 2 plies** | **97,9 ± 10,7** | **25** | **37,8 %** | **8,0 %** |

**El segundo ply parte la pérdida al medio y reduce los errores graves a un
cuarto**: es el que cierra el punto ciego de la recaptura, que a un ply queda
fuera del árbol.

### Partidas contra Stockfish (requerimientos 2.4 y 6.3, notebook 08 sección 8)

Cada motor jugó 90 partidas contra Stockfish con la fuerza acotada por
`UCI_Elo` —30 contra cada escalón de 1320, 1500 y 1700—, con 15 aperturas del
split de test jugadas con los dos colores y 50 ms por jugada para el rival. Las
360 partidas terminaron por las reglas, sin ninguna interrumpida ni con jugadas
ilegales. El Elo se ajusta contra los tres escalones a la vez, por máxima
verosimilitud (el cálculo, con fórmulas, está en la sección 8 de la 08).

| motor | victorias | tablas | derrotas | Elo (± 1σ) |
|---|---|---|---|---|
| ResNet, 1 ply | 4 % | 17 % | 79 % | 1124 ± 56 |
| Transformer, 1 ply | 0 % | 9 % | 91 % | 916 ± 90 |
| **ResNet, 2 plies** | **46 %** | **16 %** | **39 %** | **1534 ± 40** |
| **Transformer, 2 plies** | **26 %** | **18 %** | **57 %** | **1373 ± 41** |

Contra Stockfish a fuerza completa pero limitado a **un ply** —la misma búsqueda
que el motor, así que lo único distinto es la evaluación—, la ResNet sacó 0,283
puntos por partida y el Transformer 0,250 (30 partidas cada uno, sin victorias;
las tablas, casi todas por repetición).

**Sobre la escala.** Es Elo en la escala de `UCI_Elo` de Stockfish, no de
Lichess ni FIDE. Y tiene una salvedad: leído escalón por escalón, el Elo
implícito sube con el escalón (la ResNet a 2 plies da 1467, 1512 y 1617), señal
de que a 50 ms por jugada los niveles de Stockfish quedan más juntos de lo que
dicen sus etiquetas. El ± es por eso un piso de la incertidumbre del número
absoluto; la comparación entre motores medidos contra la misma escalera es más
firme.

## 4. La comparación entre arquitecturas

| profundidad | diferencia de ACPL | diferencia de Elo |
|---|---|---|
| 1 ply | 80,3 ± 23,5 a favor de la ResNet | 208 ± 106 a favor de la ResNet |
| 2 plies | 10,5 ± 15,5 a favor de la ResNet — dentro del error | 161 ± 57 a favor de la ResNet |

El empate en RMSE no se traduce en un empate en el tablero. A un ply la ResNet
juega claramente mejor en las dos medidas. A dos plies la búsqueda empareja la
pérdida media, pero no el Elo: una partida la decide el peor error, y la ResNet
sigue cometiendo menos (5,5 % contra 8,0 % de jugadas que pierden más de
300 cp). **La búsqueda achica la diferencia entre arquitecturas, pero no la
borra**, y el RMSE resultó un predictor pobre de la fuerza de juego entre dos
modelos tan parecidos.

## 5. Más datos (apéndice)

El bloque 4 concluyó que el límite lo ponía el dataset. El apéndice lo puso a
prueba entrenando la misma ResNet con 5, 10 y 30 millones de posiciones del dump
público de evaluaciones de Lichess, y midiendo esos modelos con las mismas
mediciones de la 08 sobre el mismo test:

| modelo | RMSE test | ACPL 1 ply | ACPL 2 plies | Elo 1 ply | Elo 2 plies |
|---|---|---|---|---|---|
| ResNet (memoria, 2,3 M) | 0,2511 | 176,8 | 87,4 | 1124 ± 56 | 1534 ± 40 |
| 5 M | 0,2592 | 120,4 | 70,3 | 1219 ± 48 | 1501 ± 40 |
| 10 M | 0,2369 | 104,7 | 50,2 | 1431 ± 40 | 1605 ± 41 |
| **30 M** | **0,2178** | **70,9** | **47,3** | **1511 ± 40** | **1762 ± 46** |
| 30 M, red de 16 bloques | 0,2143 | 73,9 | 49,5 | 1576 ± 40 | 1720 ± 44 |

(Elo contra los mismos tres escalones que la 08.)

- Con 10 M ya se supera a la memoria en todas las medidas; con 30 M, por unos
  230 puntos de Elo a 2 plies y casi 400 a 1 ply.
- La red el doble de profunda baja el RMSE un 1,6 % y no se distingue en el
  tablero: a este volumen la palanca es el dato, no la capacidad.
- Con 30 M y la red grande, el motor **empata con Stockfish limitado a un ply**
  (0,533 ± 0,058 puntos por partida): con la misma búsqueda, la evaluación
  aprendida iguala a la de Stockfish.

La salvedad: esos modelos no solo vieron más posiciones, vieron otras —otro
Stockfish, a otra profundidad, con diez veces más mates y más táctica—, así que
no toda la diferencia es de volumen. El detalle está en las cuatro notebooks de
[`notebooks/apendice/`](../notebooks/apendice).

## 6. Requerimientos, con su evidencia

| requerimiento | evidencia |
|---|---|
| 1.1 Pipeline de datos | `chessdl.data`, notebooks 00–02, [`pipeline.md`](pipeline.md) |
| 1.2 Formato reutilizable | Parquet en Hugging Face; test de ida y vuelta en `test_dataset_integrity.py` |
| 1.3 Balance de color | 49,68 % / 50,32 %, por construcción del muestreo |
| 1.4 Salida en [−1, 1] | 100 % del test; posiciones de referencia (08, 5.2 y 7.2) |
| 1.5 Solo jugadas legales | `test_engine.py` (enroque, captura al paso, coronación); 420 partidas sin una ilegal |
| 1.6 Búsqueda de al menos un ply | `chessdl.engine.search`; mate en uno encontrado siempre |
| 1.7 Hasta 5 s por jugada | 23 ms a 1 ply y 583 ms a 2, en el p99 (08, sección 6) |
| 2.2 Reproducibilidad | README, sección "Reproducir el trabajo" |
| 2.3 Documentación del etiquetado | [`dataset_card.md`](dataset_card.md) |
| 2.4 50 partidas o más contra Stockfish | 90 por motor, con victorias, tablas y derrotas (sección 3) |
| 3.1 Tests de jugadas especiales | `test_engine.py`, `test_encoding.py`, `test_ui.py` |
| 3.2 Tests de integridad del dataset | `test_dataset_integrity.py`, 8 chequeos en verde sobre el dataset publicado |
| 4.1 Línea de comandos | `chessdl-play`, ver el [manual de usuario](manual_usuario.md) |
| 4.2 Evaluación en centipeones | transformación inversa `400·atanh(valor)`, comparada con Stockfish (08, 7.2) |
| 5.1 Datos públicos de Lichess | dumps CC0 de database.lichess.org |
| 5.2 Uso responsable | advertencia en README, dataset card y manual |
| 6.1 Transformer (opcional) | notebooks 06 y 07 |
| 6.2 Búsqueda de más de un ply (opcional) | 2 plies dentro del presupuesto |
| 6.3 Elo aproximado (opcional) | sección 3 |

## 7. Limitaciones

- **La escala del Elo.** Los niveles de Stockfish a 50 ms por jugada no están
  tan separados como dicen sus etiquetas, así que el Elo absoluto es menos firme
  que el orden entre motores.
- **Búsqueda corta.** Sin poda alfa-beta ni búsqueda de capturas, la profundidad
  2 es el máximo que entra en 5 segundos; las tácticas de tres jugadas o más y
  los finales que exigen un plan quedan fuera de su alcance.
- **Posiciones igualadas.** Es donde la evaluación más falla (60 % de acuerdo de
  signo) y donde el motor decide sus jugadas.
- **Una sola semilla por configuración.** Diferencias de RMSE por debajo del 1 %
  entre modelos no se pueden distinguir del ruido de entrenamiento.
