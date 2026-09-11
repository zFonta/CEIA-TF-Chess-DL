# Sistema de evaluación de posiciones de ajedrez y selección de jugadas óptimas mediante aprendizaje profundo

Trabajo Final — Carrera de Especialización en Inteligencia Artificial (FIUBA)

- **Autor:** Act. Federico Santiago Fontanari
- **Director:** Esp. Ing. Gerardo Alexis Vilcamiza Espinoza

## Estado del proyecto

Este repositorio implementa el **bloque 3 del WBS: pipeline de datos**. Genera un
dataset reproducible de pares *(posición, evaluación)* etiquetados con Stockfish,
a partir de partidas públicas de Lichess.

| Bloque del WBS | Estado |
|---|---|
| 3. Pipeline de datos | **Completo — dataset generado y publicado** |
| 4. Red neuronal | Pendiente |
| 5. Motor de juego | Pendiente |
| 6. Evaluación del sistema | Pendiente |

Requerimientos del plan cubiertos: **1.1, 1.2, 1.3, 2.2, 2.3, 3.2, 5.1**, y la
transformación inversa que necesita el 4.2.

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

Las tres notebooks del repositorio están versionadas **con la salida de esa
corrida**, así que los números se pueden auditar sin volver a ejecutar nada.

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

## Reproducir el dataset (requerimiento 2.2)

El entorno de ejecución del proyecto es **Google Colab Pro**, y las notebooks son
el punto de entrada. Se ejecutan en orden:

| Notebook | Para qué |
|---|---|
| `notebooks/00_survey_dumps.ipynb` | Medir cuántas partidas pasan el filtro **antes** de gastar cómputo |
| `notebooks/01_build_dataset.ipynb` | Construir el dataset (reanudable) y subirlo a Hugging Face |
| `notebooks/02_dataset_eda.ipynb` | Análisis exploratorio, verificación del espejado y de la transformación inversa (WBS 3.5) |

Todo el código del proyecto se ejecuta desde las notebooks: los módulos se
importan directamente, los tests corren en una celda `!{sys.executable} -m pytest`,
y los dos comandos de línea de comando tienen su celda equivalente. La única
excepción es `tests/fixtures/make_fixture.py`, una herramienta de desarrollo que
regenera el PGN de prueba y solo se corre si se quiere cambiar su contenido.

> **Usar un runtime de CPU, no de GPU.** El etiquetado con Stockfish es puro CPU
> y los runtimes con GPU de Colab traen *menos* vCPUs: elegir GPU es más lento
> acá y además consume cuota que conviene reservar para el entrenamiento.

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
| Shards etiquetados | `zFonta/ceia-chess-eval` — el entregable |
| Extracto PGN filtrado | `zFonta/ceia-chess-work` |
| Estado de reanudación | `zFonta/ceia-chess-work` |
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
├── hf.py              # subida/bajada de shards a Hugging Face
├── data/
│   ├── lichess.py     # streaming de los dumps, extracción filtrada
│   ├── pgn.py         # parseo de cabeceras y filtro de partidas
│   ├── sampling.py    # muestreo de posiciones balanceado por turno
│   ├── labeling.py    # etiquetado con Stockfish en paralelo
│   ├── schema.py      # esquema Parquet y E/S de shards
│   ├── state.py       # estado de reanudación y deduplicación
│   ├── pipeline.py    # orquestación de punta a punta
│   └── validate.py    # chequeos de integridad
└── scripts/           # interfaces de línea de comando
```

## Tests

`pytest` descubre solo los archivos `tests/test_*.py` y ejecuta cada función
`test_*`. No hay que importar ni invocar nada a mano.

```bash
pytest -q                          # los 204 tests, ~30 segundos
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

Los tests corren sin red. Los que necesitan Stockfish se saltean solos si no está
instalado; el resto usa el fixture PGN versionado en `tests/fixtures/`.

Lo que cubren, en orden de importancia:

- **`test_encoding.py`** — el espejado del tablero: invarianza, derechos de
  enroque, captura al paso, promoción, y que la codificación de un board vivo
  coincida con la de su FEN.
- **`test_pipeline.py`** — la construcción completa contra el fixture con
  Stockfish real, incluida la reanudación tras una interrupción.
- **`test_dataset_integrity.py`** — que cada chequeo del requerimiento 3.2 falle
  cuando se le inyecta el problema que dice detectar.
- **`test_notebooks.py`** — que las celdas de entorno de las tres notebooks no se
  desincronicen. Cubre las fallas que ya costaron una sesión de Colab: instalar
  sin el extra `dev`, no poner `src/` en `sys.path`, tapar un `git pull` fallido
  con `check=False`, o llamar a `!python` en vez de `!{sys.executable}`.
- **`test_normalize.py`**, **`test_sampling.py`**, **`test_pgn_filter.py`**.

## Fuente de datos y licencia

Las partidas provienen de la base pública de [Lichess](https://database.lichess.org),
distribuida bajo **CC0** (requerimiento 5.1). El código de este repositorio está
bajo licencia MIT.

Este sistema **no debe usarse para asistir a jugadores en partidas competitivas
en línea**: constituiría una violación de los términos de servicio de las
plataformas de ajedrez (requerimiento 5.2).
