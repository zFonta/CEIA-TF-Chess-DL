# Manual de usuario

Cómo usar el motor y cómo leer lo que dice. Está escrito para quien sabe jugar
al ajedrez y no necesita saber programar: todo lo que sigue se hace desde el
navegador.

## Qué es y qué no es

Es un motor de ajedrez cuya **evaluación de las posiciones la aprendió una red
neuronal**, a partir de 2,5 millones de posiciones de partidas de Lichess
evaluadas por Stockfish. Para elegir una jugada prueba todas las legales, mira
una o dos jugadas hacia adelante y se queda con la que deja la mejor posición
según la red.

Juega más o menos como un aficionado de club: alrededor de **1.500 de Elo** con
dos jugadas de búsqueda, en la escala de Stockfish (no es directamente un Elo de
Lichess ni FIDE). Juega aperturas razonables, evalúa bien el material y las
amenazas inmediatas, y se le escapan las tácticas largas y los planes de muchas
jugadas, como ganar un final de torre contra rey solo.

> **No debe usarse para asistir a jugadores en partidas en línea.** Va contra
> los términos de servicio de las plataformas de ajedrez (requerimiento 5.2).

## Jugar contra el motor, sin instalar nada

1. Abrir la notebook
   [`09_jugar_contra_el_motor.ipynb`](../notebooks/09_jugar_contra_el_motor.ipynb)
   en GitHub y apretar el botón **Open in Colab** de arriba. Hace falta una
   cuenta de Google; no hace falta cuenta de Hugging Face, porque los modelos
   son públicos.
2. En Colab: menú **Entorno de ejecución → Ejecutar todas**. La primera vez
   tarda un par de minutos: instala el paquete, baja Stockfish y baja los dos
   modelos. Si Colab pregunta si confiar en la notebook, aceptar.
3. Al final de la notebook aparece el tablero.

Funciona igual sin GPU: una jugada a dos plies tarda décimas de segundo.

### El tablero

- **Mover:** click en la pieza y después click en el casillero. Los destinos
  legales aparecen marcados con un punto y las capturas con el casillero en
  rojo.
- **Enrocar:** click en el rey y después en la torre.
- **Coronar:** el tablero pregunta qué pieza se quiere; no asume una dama.
- **Botones:** *Nueva* empieza otra partida; *Deshacer* vuelve atrás tu jugada y
  la del motor; *Girar* da vuelta el tablero; *Sugerir* muestra la jugada que
  haría el motor en tu lugar, sin jugarla; *Jugar por mí* la juega.
- **Color:** *Juego con blancas* o *Juego con negras*, y después *Nueva*. Con
  negras abre el motor.
- **Posiciones:** los botones de posición cargan casos preparados (un mate en
  uno, una dama colgada, un final de torre, la posición inicial). También se
  puede pegar cualquier posición en formato **FEN** y apretar *Cargar*.
- **Modelo:** cambia entre las dos redes —ResNet y Transformer— en cualquier
  momento de la partida.
- **Profundidad:** cuántas jugadas mira hacia adelante. **1 ply** es su propia
  jugada; **2 plies** suma la respuesta del rival y juega bastante mejor (unos
  400 puntos de Elo más). **3 plies** existe para mirar, pero tarda entre 9 y 15
  segundos por jugada y queda fuera del límite de 5 segundos del trabajo.

## Cómo leer la evaluación

### El número

El motor da la evaluación de dos maneras equivalentes, **siempre desde el punto
de vista de las blancas**:

- **Un valor entre −1 y +1.** +1 es "las blancas ganan", −1 "las negras ganan" y
  0 "pareja".
- **Centipeones** (cp), la escala habitual de los motores: 100 cp equivalen
  aproximadamente a un peón de ventaja. Positivo es ventaja blanca.

Los dos se convierten uno en otro, y la relación no es lineal: cerca de cero un
poco de valor son muchos centipeones, y cerca de ±1 hace falta mucha ventaja
para mover el valor.

| ventaja de las blancas | centipeones | valor |
|---|---|---|
| pareja | 0 | 0,00 |
| medio peón | +50 | +0,12 |
| un peón | +100 | +0,24 |
| dos peones | +200 | +0,46 |
| una pieza menor | +300 | +0,64 |
| una torre | +500 | +0,85 |
| una dama | +900 | +0,98 |
| decisiva | +2.000 | +1,00 |

Para las negras, lo mismo con signo negativo. Las evaluaciones se recortan en
±2.000 cp: más allá de eso todo es "gana", y el motor no distingue entre ganar
con comodidad y ganar con más comodidad.

### Las dos barras

Al costado del tablero hay **dos barras**: la evaluación de la red y la de
Stockfish a profundidad 12, en la misma escala. Stockfish está como referencia,
no como rival: es la evaluación que la red aprendió a imitar. **La distancia
entre las dos barras es el error de la red en esa posición.**

### "Lo que ve el motor"

El panel lista las cinco jugadas que el motor considera mejores, cada una con su
valor. Ahí se ve el criterio y no solo la decisión: si dos jugadas están casi
empatadas, el motor dudó; si la elegida está muy por encima de las demás, la vio
clara.

Un **#3** en vez de un número quiere decir que esa jugada da mate en tres. Los
mates no los calcula la red sino las reglas: cuando una jugada termina la
partida, el motor lo sabe con certeza y no le pregunta a la red.

### Por qué la red y Stockfish pueden no coincidir

- **Tácticas más allá del horizonte.** Con uno o dos plies el motor no ve una
  combinación de tres jugadas. La red a veces la "intuye" mirando el tablero
  quieto, y a veces no.
- **Mates.** Stockfish anuncia "mate en N"; la red solo dice "muy ganado", con un
  valor cerca de ±1.
- **Finales técnicos.** La red sabe que rey y torre contra rey está ganado, pero
  ganar exige un plan de muchas jugadas, y con dos plies de búsqueda el motor no
  lo encuentra: puede dar vueltas hasta que la repetición o la regla de las 50
  jugadas terminen la partida en tablas.

## Desde la línea de comandos (opcional)

Para quien tenga Python instalado. Desde la carpeta del repositorio:

```bash
pip install -e ".[train]"      # el paquete y PyTorch, que es lo que corre la red

# Analizar una posicion: jugada recomendada y evaluacion (requerimiento 4.1)
chessdl-play --run-name campana2-warmup --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" --depth 2
```

La salida dice quién juega, la **jugada elegida**, la **evaluación** —valor y
centipeones, desde las blancas— y el tiempo que tardó. Debajo va una tabla con
las mejores jugadas; esa tabla está **desde el punto de vista del que mueve**
(positivo es bueno para quien juega), porque es lo que el motor compara para
elegir.

Opciones útiles:

| opción | qué hace |
|---|---|
| `--fen "<FEN>"` | la posición a analizar; sin ella, la inicial |
| `--depth 1` o `--depth 2` | plies de búsqueda; 1 por defecto, 2 juega mejor |
| `--top N` | cuántas jugadas mostrar en la tabla |
| `--self-play N` | el motor juega N jugadas contra sí mismo y muestra la partida |
| `--run-name transformer-campana2-lotes-chicos` | usar el transformer en vez de la ResNet |

## Problemas comunes

| síntoma | qué pasa y qué hacer |
|---|---|
| El tablero no aparece | Alguna celda anterior falló. Revisar la primera celda con error, y volver a **Ejecutar todas** |
| Colab dice que la sesión se desconectó | Volver a ejecutar todas las celdas: la instalación es idempotente y retoma |
| `Hace falta --checkpoint o --run-name` | La línea de comandos necesita saber qué modelo usar; agregar `--run-name campana2-warmup` |
| El motor no gana un final ganado | Es una limitación conocida de la búsqueda corta (ver arriba), no un error |
