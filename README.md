# Sistema de evaluación de posiciones de ajedrez y selección de jugadas óptimas mediante aprendizaje profundo

Trabajo Final — Carrera de Especialización en Inteligencia Artificial (FIUBA)

- **Autor:** Act. Federico Santiago Fontanari
- **Director:** Esp. Ing. Gerardo Alexis Vilcamiza Espinoza

## Estado del proyecto

Este repositorio implementa el trabajo completo: el pipeline que genera un
dataset reproducible de pares *(posición, evaluación)* etiquetados con Stockfish
a partir de partidas públicas de Lichess, las dos redes que aprenden a reproducir
esa evaluación, y el motor que las usa para elegir jugadas.

| Bloque del WBS | Estado |
|---|---|
| 3. Pipeline de datos | **Completo** — dataset generado y publicado |
| 4. Red neuronal | **Completo** — ResNet y transformer entrenados y comparados |
| 5. Motor de juego | **Completo** — búsqueda negamax, CLI y tiempos medidos |
| 6. Evaluación del sistema | **Completo** — pérdida en centipeones y partidas contra Stockfish |

Todos los bloques están **corridos**: los números de este README salen de las
notebooks versionadas con su salida.

Requerimientos del plan cubiertos: **1.1 a 1.4, 1.6, 1.7, 2.2, 2.3, 3.1, 3.2,
4.2, 5.1, 5.2 y 6.1**.

El **1.7** (5 segundos por jugada) quedó medido y no estimado: **10 ms** en el
percentil 90 de partidas reales sobre Colab T4 con búsqueda de un nivel, y
**285 ms** con dos. El número depende del hardware, así que va siempre con él al
lado.

### El dataset generado

Corrida completa en Colab Pro, publicada en
[`zFonta/ceia-chess-eval`](https://huggingface.co/datasets/zFonta/ceia-chess-eval):

| | |
|---|---|
| Posiciones | **2.552.804** en 157 shards |
| Partidas | 772.797, del dump `2025-06` de Lichess |
| Etiquetas | Stockfish 17.1 a profundidad 12 |
| Balance de color | 49,68 % / 50,32 % |
| Chequeos de integridad | 8 de 8 en verde, cero duplicados |

El detalle completo —composición, distribuciones, validación manual del
etiquetado y limitaciones conocidas— está en
[`docs/dataset_card.md`](docs/dataset_card.md); lo que costó generarlo y qué
esperar al volver a correrlo, en [`docs/pipeline.md`](docs/pipeline.md).

Las notebooks del repositorio están versionadas **con la salida de sus corridas
reales**, así que todos los números de este README se pueden auditar sin volver a
ejecutar nada.

### Los modelos entrenados

Pesos, métricas e historiales en
[`zFonta/ceia-chess-models`](https://huggingface.co/zFonta/ceia-chess-models).
Todo sobre el split de **test**, que se toca una sola vez por campaña:

| | test RMSE | R² | MAE (cp) | signo | ρ | parámetros |
|---|---|---|---|---|---|---|
| Piso: media constante | 0,4886 | 0,000 | — | — | — | — |
| Piso: material lineal | 0,3973 | 0,339 | — | — | — | — |
| ResNet campaña 1 | 0,2609 | 0,715 | 110,5 | 86,74 % | 0,8240 | 2.913.345 |
| **ResNet campaña 2** | **0,2511** | **0,736** | **105,8** | **87,80 %** | **0,8385** | 2.913.345 |
| Transformer campaña 1 | 0,2978 | 0,629 | 125,5 | 82,68 % | 0,7592 | 2.735.361 |
| **Transformer campaña 2** | **0,2531** | **0,732** | **109,0** | **86,80 %** | **0,8315** | 2.735.361 |

**Las dos arquitecturas empatan.** Con presupuestos de parámetros equiparados, la
diferencia final es del 0,80 % — menos de lo que se esperaría entre dos semillas
de la misma configuración. Las dos mejoran el piso de material un 36 %, y las dos
alcanzan su mejor época a los dos tercios del presupuesto y empiezan a memorizar
después.

Que dos arquitecturas con priors tan distintos —la convolución trae la localidad
de fábrica, la atención tiene que aprenderla— lleguen al mismo número y con la
misma forma de curva es el resultado principal del bloque 4: **el techo lo pone
el dataset, no cómo se lee el tablero.**

El razonamiento completo, los diagnósticos de sobre y subajuste y lo que se
probó en cada campaña están en [`docs/modelado.md`](docs/modelado.md).

### El motor jugando

El RMSE mide cuánto se parece la red a Stockfish; la **pérdida media en
centipeones** mide cuánto juega. Sobre 400 posiciones de test, comparando cada
jugada elegida contra la mejor según Stockfish a profundidad 12:

| motor | ACPL | acuerdo con SF | errores > 300 cp | ms/jugada (p90) |
|---|---|---|---|---|
| ResNet, 1 ply | 176,8 ± 14,4 | 30,5 % | 21,0 % | 10 |
| Transformer, 1 ply | 257,1 ± 18,6 | 28,5 % | 34,2 % | 11 |
| **ResNet, 2 plies** | **87,4 ± 11,2** | **39,0 %** | **5,5 %** | 285 |
| **Transformer, 2 plies** | **97,9 ± 10,7** | **37,8 %** | **8,0 %** | 447 |

**Un ply más parte la pérdida al medio y reduce los errores graves a un cuarto**,
y sigue 18 veces por debajo del presupuesto del requerimiento 1.7. Es el ply que
cierra el punto ciego de la recaptura: a un nivel el árbol termina en la jugada
propia y la respuesta del rival no está en él.

Y la comparación entre arquitecturas **se da vuelta con la profundidad**: a 1 ply
la ResNet le saca 80,3 ± 23,5 cp —una diferencia clara, que el empate del 0,80 %
en RMSE no anticipaba—, y a 2 plies la brecha cae a 10,5 ± 15,5, dentro del error.
La búsqueda rescata los errores locales de la red más ruidosa, así que *"cuál
arquitectura es mejor"* depende de cuánta búsqueda haya encima.

Profundidad 3 **no entra** en el presupuesto: 9,7 s en el percentil 90 sobre T4.

## Qué hace el pipeline

```
Lichess (dumps mensuales, CC0)
    │  streaming zstd, sin descargar el archivo entero
    ▼
Filtro por ELO (≥2200 a ambos) y control de tiempo (Blitz/Rapid/Classical)
    ▼
Extracto PGN filtrado  ──────────────► se genera una sola vez, se reutiliza
    ▼
Muestreo de 4 posiciones por partida (2 con blancas al turno, 2 con negras)
    ▼
Etiquetado con Stockfish a profundidad fija
    ▼
Normalización cp → [-1, 1]  con  value = tanh(cp / 400)
    ▼
Shards Parquet ──► Hugging Face Datasets
```

Detalle completo en [`docs/pipeline.md`](docs/pipeline.md); las decisiones de
diseño y los parámetros exactos, en [`docs/dataset_card.md`](docs/dataset_card.md).

## Reproducir el trabajo (requerimiento 2.2)

El entorno de ejecución del proyecto es **Google Colab Pro**, y las notebooks son
el punto de entrada. Se ejecutan en orden.

### Bloque 3 — pipeline de datos · **runtime de CPU**

| Notebook | Para qué |
|---|---|
| `00_survey_dumps.ipynb` | Medir cuántas partidas pasan el filtro **antes** de gastar cómputo |
| `01_build_dataset.ipynb` | Construir el dataset (reanudable) y subirlo a Hugging Face |
| `02_dataset_eda.ipynb` | Análisis exploratorio, verificación del espejado y de la transformación inversa (WBS 3.5) |

> **Acá conviene CPU, no GPU.** El etiquetado con Stockfish es puro CPU y los
> runtimes con GPU de Colab traen *menos* vCPUs: elegir GPU es más lento y además
> consume cuota que conviene reservar para el entrenamiento.

### Bloque 4 — redes · **runtime de GPU (T4)**

| Notebook | Para qué |
|---|---|
| `03_train_resnet.ipynb` | Entorno de entrenamiento y arquitectura residual (WBS 4.1 y 4.2) |
| `04_train_campaign.ipynb` | Función de pérdida y primera campaña de la ResNet (4.3 y 4.4) |
| `05_hyperparameters.ipynb` | Barrido de 5 brazos y segunda campaña de la ResNet (4.5 y 4.6) |
| `06_train_transformer.ipynb` | Arquitectura transformer y su primera campaña (4.8) |
| `07_transformer_tuning.ipynb` | Barrido de 3 brazos y campaña final del transformer (4.8) |
| `08_motor_y_partidas.ipynb` | El motor, el tiempo por jugada y las partidas contra Stockfish (bloques 5 y 6) |
| `09_jugar_contra_el_motor.ipynb` | Tablero interactivo para jugar contra las dos redes |

Las notebooks del bloque 4 **no dependen de volver a correr las del bloque 3**:
bajan el dataset ya publicado del Hub. Y cada campaña es reanudable —el estado
completo de entrenamiento se sube al Hub después de cada época—, así que una
desconexión de Colab cuesta una época, no la campaña.

Todo el código del proyecto se ejecuta desde las notebooks: los módulos se
importan directamente, los tests corren en una celda `!{sys.executable} -m pytest`,
y los dos comandos de línea de comando tienen su celda equivalente. La única
excepción es `tests/fixtures/make_fixture.py`, una herramienta de desarrollo que
regenera el PGN de prueba y solo se corre si se quiere cambiar su contenido.

### Ejecución local

```bash
git clone https://github.com/zFonta/CEIA-TF-Chess-DL.git
cd CEIA-TF-Chess-DL

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

bash scripts/setup_stockfish.sh          # instala una versión fija de Stockfish
export PATH="$PWD/bin:$PATH"
```

Relevar el rendimiento del filtro sin etiquetar nada:

```bash
python -m chessdl.scripts.build_dataset --survey --max-scanned 1000000
```

Construir el dataset:

```bash
python -m chessdl.scripts.build_dataset            # completo, reanudable
python -m chessdl.scripts.build_dataset --max-shards 1 --no-push   # piloto local
```

Validar la integridad (requerimiento 3.2):

```bash
python -m chessdl.scripts.validate_dataset --shards-dir <dir> --stats
```

Jugar o analizar con el motor (requerimientos 1.6, 1.7 y 4.2):

```bash
# Analizar una posicion: la jugada elegida y las alternativas, en centipeones
python -m chessdl.scripts.play --run-name campana2-warmup --fen "<FEN>"

# Una partida del motor contra si mismo, con el tiempo por jugada medido
python -m chessdl.scripts.play --run-name campana2-warmup --self-play 40
```

Sin `--run-name` ni `--checkpoint` el comando **se niega a jugar**: una red sin
entrenar devuelve jugadas legales y parecería estar funcionando.

### Configuración

Todos los parámetros viven en [`configs/dataset_v1.yaml`](configs/dataset_v1.yaml),
que es la única fuente de verdad y se versiona junto con el código. Los overrides
de línea de comando (`--engine`, `--workers`, `--shards-dir`, …) son para pruebas;
lo que define una versión del dataset es el YAML.

### Credenciales

El token de Hugging Face se lee de la variable de entorno `HF_TOKEN` o del panel
de *Secrets* de Colab, y necesita permiso de **escritura**. El namespace sale de
`HF_NAMESPACE` (por defecto `zFonta`). **Nunca se guardan credenciales en el
repositorio.**

## Dónde vive cada cosa

Todo lo durable está en Hugging Face; el disco local es solo un cache
descartable. **No se usa Google Drive**: no hay unidad que montar, y una corrida
se puede continuar desde cualquier máquina.

| Qué | Dónde |
|---|---|
| Shards etiquetados | `zFonta/ceia-chess-eval` — el entregable del bloque 3 |
| Pesos, métricas e historiales | `zFonta/ceia-chess-models` — el entregable del bloque 4 |
| Extracto PGN filtrado | `zFonta/ceia-chess-work` |
| Estado de reanudación | `zFonta/ceia-chess-work` |
| Caché de tensores | local, descartable — se reconstruye desde los FEN |
| Cache de trabajo | local, descartable |

La deduplicación no se almacena: se reconstruye leyendo la columna `pos_key` de
los shards publicados, así que el dataset es su propio registro de lo que
contiene.

## Cargar el dataset

```python
from chessdl import hf
from chessdl.data import schema

directorio = hf.download_dataset("zFonta/ceia-chess-eval", "./data")
tabla = schema.read_dataset(schema.shard_paths(directorio))
```

Y para codificar una posición como tensor de entrada de la red:

```python
import chess
from chessdl.encoding import board_to_tensor   # (18, 8, 8) float32

tensor = board_to_tensor(chess.Board(tabla["fen"][0].as_py()))
```

## Evaluar una posición con un modelo entrenado

```python
import chess, torch
from chessdl import hf
from chessdl.encoding import board_to_tensor
from chessdl.models.resnet import ChessResNet, ResNetConfig
from chessdl.normalize import value_to_cp
from chessdl.training.checkpoint import BEST_NAME, HubCheckpoints, load_checkpoint

checkpoints = HubCheckpoints(
    repo_id="zFonta/ceia-chess-models",
    run_name="campana2-warmup",          # la mejor ResNet
    local_dir="./checkpoints",
    token=hf.get_token(),
)
modelo = ChessResNet(ResNetConfig(channels=128, blocks=8))
load_checkpoint(checkpoints.fetch(BEST_NAME), modelo)
modelo.eval()

tablero = chess.Board()
entrada = torch.from_numpy(board_to_tensor(tablero)).unsqueeze(0)
with torch.no_grad():
    value_stm = float(modelo(entrada))

# La red predice en perspectiva del jugador al turno (requerimiento 1.4 sobre
# `value_stm`); el cambio de signo la lleva a perspectiva de las blancas.
value_white = value_stm if tablero.turn == chess.WHITE else -value_stm
print(f"{value_white:+.4f}  ({value_to_cp(value_white):+.0f} centipeones)")
```

Para el transformer, cambiar `ChessResNet`/`ResNetConfig` por
`ChessTransformer`/`TransformerConfig` y el `run_name` por
`transformer-campana2-lotes-chicos`. Las dos arquitecturas exponen la misma
interfaz —entrada `(batch, 18, 8, 8)`, salida `(batch,)` en [−1, 1]—, así que el
resto del código no cambia.

## Jugar contra el motor

La notebook [`09_jugar_contra_el_motor.ipynb`](notebooks/09_jugar_contra_el_motor.ipynb)
levanta un tablero clickeable —click en la pieza, click en el casillero— contra
cualquiera de las dos redes, y muestra mientras tanto la evaluación de la
posición y las cinco jugadas que el motor rankeó más alto, con su valor.

```python
from chessdl import hf
from chessdl.engine.evaluator import Evaluator
from chessdl.engine.loader import load_from_hub
from chessdl.ui import PlayUI

modelo = load_from_hub("zFonta/ceia-chess-models", "campana2-warmup", token=hf.get_token())
PlayUI(Evaluator(modelo), depth=2)      # la última expresión de la celda lo dibuja
```

No agrega ninguna medición: las de la memoria salen de la notebook 08. Lo que
agrega es poder **verificar a mano**, sobre posiciones elegidas por quien lee,
las tres cosas que esa notebook afirma — que el mate se encuentra por reglas y
no por la red, que un ply no ve la recaptura y dos sí, y que las dos
arquitecturas se parecen mucho más de lo que sus nombres sugieren.

Necesita `ipywidgets` (`pip install -e ".[ui]"`, ya instalado en Colab). La
lógica de los clicks vive en `chessdl.ui.game`, sin widgets: es donde un tablero
interactivo falla de verdad —una captura leída como reselección, una coronación
que asume una dama, un *deshacer* que devuelve el turno al lado equivocado—, y
nada de eso se ve en una captura de pantalla ni se puede testear moviendo
widgets.

Para una jugada suelta desde la terminal, sin tablero, está `chessdl-play`.

## Decisión de diseño central: perspectiva del jugador al turno

La red **siempre ve el tablero como si le tocara jugar a las blancas**. Cuando
mueven las negras, el tablero se espeja verticalmente y se intercambian los
colores, de modo que *mis* piezas caen siempre en los planos 0–5 y las del rival
en los planos 6–11. La red no tiene que aprender dos representaciones simétricas
del mismo concepto.

Como consecuencia, el dataset guarda la etiqueta en **los dos puntos de vista**:

- `value_stm` — perspectiva del jugador al turno; **es el target de entrenamiento**.
- `value_white` — perspectiva de las blancas; es la escala de los requerimientos
  1.4 y 4.2, la que reportan el motor y la CLI.

Las dos son el mismo número salvo el signo, y hay un chequeo de integridad que
verifica esa relación en todas las filas del dataset.

## Estructura

```
src/chessdl/
├── config.py          # configuración tipada, cargada del YAML
├── encoding.py        # posición → tensor (18,8,8), espejado según el turno
├── normalize.py       # cp ↔ [-1,1], directa e inversa
├── viz.py             # estilo de figuras para el análisis y la memoria
├── colab.py           # entorno Colab: secretos y chequeo de runtime
├── hf.py              # subida/bajada a Hugging Face (datasets y modelos)
├── data/                       # bloque 3
│   ├── lichess.py     # streaming de los dumps, extracción filtrada
│   ├── pgn.py         # parseo de cabeceras y filtro de partidas
│   ├── sampling.py    # muestreo de posiciones balanceado por turno
│   ├── labeling.py    # etiquetado con Stockfish en paralelo
│   ├── schema.py      # esquema Parquet y E/S de shards
│   ├── state.py       # estado de reanudación y deduplicación
│   ├── pipeline.py    # orquestación de punta a punta
│   └── validate.py    # chequeos de integridad
├── models/                     # bloque 4: las arquitecturas
│   ├── resnet.py      # ResNet estilo AlphaZero, 3x3 sin reducir el tablero
│   └── transformer.py # encoder sobre 64 tokens, uno por casilla
├── engine/                     # bloques 5 y 6: el motor
│   ├── evaluator.py   # evalúa lotes de posiciones; terminales desde las reglas
│   ├── search.py      # negamax a profundidad N sobre la evaluación
│   ├── loader.py      # reconstruye el modelo desde el model_config del checkpoint
│   └── match.py       # partidas contra Stockfish, Elo y pérdida en centipeones
├── ui/                         # tablero interactivo para la notebook
│   ├── game.py        # qué significa cada click; sin widgets, y es la parte testeada
│   └── board.py       # lo dibuja con ipywidgets
├── training/                   # bloque 4: cómo se entrenan
│   ├── split.py       # partición POR PARTIDA, con hash estable
│   ├── cache.py       # caché uint8 de tensores (2,9 GB en vez de 11,8)
│   ├── dataset.py     # Dataset de PyTorch sobre el caché
│   ├── baselines.py   # pisos de media y de material
│   ├── loss.py        # MSE y Huber
│   ├── metrics.py     # RMSE, MAE en cp, acuerdo de signo, Spearman
│   ├── loop.py        # entrenamiento reanudable, con calentamiento y recorte
│   ├── checkpoint.py  # estado completo al Hub después de cada época
│   └── experiments.py # barridos de hiperparámetros de las dos arquitecturas
└── scripts/           # interfaces de línea de comando
```

## Tests

`pytest` descubre solo los archivos `tests/test_*.py` y ejecuta cada función
`test_*`. No hay que importar ni invocar nada a mano.

```bash
pytest -q                          # los 539 tests, ~40 segundos
pytest tests/test_encoding.py -v   # un archivo, mostrando test por test
pytest -k mirror -v                # solo los que matcheen ese texto en el nombre
pytest --collect-only -q           # listarlos sin ejecutarlos
```

**Desde Colab**, en una celda, con `!` adelante:

```
!{sys.executable} -m pytest -q
```

La notebook `01_build_dataset.ipynb` ya trae esa celda en su sección 2: conviene
correrla antes de lanzar el etiquetado, y su salida sirve como evidencia de los
requerimientos de testing (3.1 y 3.2) para la memoria. Para que `pytest` esté
disponible hay que instalar con el extra `dev` (`pip install -e ".[dev]"`), que
es lo que hace la celda de entorno de las notebooks.

Los tests corren sin red y sin GPU. Los que necesitan Stockfish se saltean solos
si no está instalado, y los de entrenamiento si no está el extra `train`; el
resto usa el fixture PGN versionado en `tests/fixtures/`.

Lo que cubren, en orden de importancia:

- **`test_encoding.py`** — el espejado del tablero: invarianza, derechos de
  enroque, captura al paso, promoción, y que la codificación de un board vivo
  coincida con la de su FEN.
- **`test_split.py`** — que la partición sea **por partida** y no por posición.
  Es la decisión metodológica más importante del bloque 4: hacerla mal infla la
  validación y no se detecta hasta que el motor juega peor que sus métricas.
- **`test_pipeline.py`** — la construcción completa contra el fixture con
  Stockfish real, incluida la reanudación tras una interrupción.
- **`test_training_loop.py`** — que una corrida reanudada llegue al mismo lugar
  que una ininterrumpida: no alcanza con recargar los pesos, tienen que volver
  el optimizador, el schedule, el escalador de precisión mixta y el orden de
  barajado de cada época.
- **`test_dataset_integrity.py`** — que cada chequeo del requerimiento 3.2 falle
  cuando se le inyecta el problema que dice detectar.
- **`test_resnet.py`** y **`test_transformer.py`** — los modos de falla que no
  dan error: una red que corre pero no aprende, un tensor de tokens que perdió
  la casilla, embeddings posicionales inertes.
- **`test_notebooks.py`** — que las celdas de entorno de las notebooks no se
  desincronicen. Cubre las fallas que ya costaron una sesión de Colab: instalar
  sin el extra `dev`, no poner `src/` en `sys.path`, tapar un `git pull` fallido
  con `check=False`, o llamar a `!python` en vez de `!{sys.executable}`.
- **`test_normalize.py`**, **`test_metrics.py`**, **`test_baselines.py`**,
  **`test_training_cache.py`**, **`test_experiments.py`**,
  **`test_sampling.py`**, **`test_pgn_filter.py`**.

## Fuente de datos y licencia

Las partidas provienen de la base pública de [Lichess](https://database.lichess.org),
distribuida bajo **CC0** (requerimiento 5.1). El código de este repositorio está
bajo licencia MIT.

Este sistema **no debe usarse para asistir a jugadores en partidas competitivas
en línea**: constituiría una violación de los términos de servicio de las
plataformas de ajedrez (requerimiento 5.2).
