# Compresor de Audio Lossless (LPC + Rice Coding)

Implementación en Python de un **códec de audio sin pérdidas (lossless)**. El sistema utiliza **Codificación Predictiva Lineal (LPC)** para modelar la señal espectralmente y **Codificación Rice** (Golomb-Rice) con adaptación dinámica de parámetros para comprimir el residuo.

## Descripción del Proyecto

El proyecto tiene como finalidad comprimir una señal PCM sin pérdidas estilo FLAC. La versión 3 (`v3`) introduce **procesamiento paralelo** para acelerar tanto la codificación como la decodificación.

### Flujo de Procesamiento (Encoder)

1.  **Tramado (Framing):** Segmentación entramas (por defecto 4096 muestras).
2.  **Análisis LPC (Levinson-Durbin):** Para cada trama, se calcula la autocorrelación y se utiliza el algoritmo de Levinson-Durbin para encontrar los coeficientes óptimos del filtro predictor (orden configurable, por defecto 10).
3.  **Cálculo del Residuo:** Se predice la señal actual $\hat{x}[n]$ mediante la combinación lineal de muestras pasadas y los coeficientes LPC. La diferencia con la señal real es el residuo:
    $$e[n] = x[n] - \text{round}(\hat{x}[n])$$
4.  **Codificación de Entropía (Rice):**
    - **Zigzag Encoding:** Convierte el residuo (con signo) a enteros positivos para optimizar la codificación ($0 \to 0, -1 \to 1, 1 \to 2...$).
    - **Estimación de K:** Se busca el parámetro $k$ óptimo probando valores en un rango (0-16) y seleccionando aquel que minimiza el tamaño total en bits de la trama codificada.
    - **Rice Coding:** Se genera el bitstream comprimido separando el valor en cociente (unario) y resto (binario).
5.  **Empaquetado:** Se guarda un archivo binario (`.bin`) que contiene las cabeceras globales, y para cada trama: su metadata ($k$, padding, longitud), los coeficientes LPC y el bitstream comprimido.

### Flujo de Decodificación (Decoder)

1.  **Lectura de Tramas:** Se extraen los parámetros $k$ y los coeficientes LPC de cada bloque.
2.  **Decodificación Rice y Zigzag:** Se recupera el residuo original $e[n]$.
3.  **Síntesis LPC:** Se reconstruye la señal sumando el residuo a la predicción generada por los coeficientes recuperados:
    $$x[n] = e[n] + \text{round}(\hat{x}[n])$$

## Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `encoder_v3.py` | **(Nuevo)** Versión paralelizada del encoder. Utiliza `ProcessPoolExecutor` para procesar tramas simultáneamente, mejorando significativamente el rendimiento. Genera `encoded_v3.bin`. |
| `decoder_v3.py` | **(Nuevo)** Versión paralelizada del decoder. Lee `encoded_v3.bin` y reconstruye el audio en `Decoded_Audio_v3.wav`. |
| `compare.py` | Script de utilidad para verificar que la compresión es realmente lossless, comparando muestra a muestra el audio original con el decodificado. |
| `encoder_v2.py` | Versión secuencial del encoder (legacy). |
| `decoder_v2.py` | Versión secuencial del decoder (legacy). |

## Configuración del Algoritmo

El sistema permite ajustar los siguientes parámetros en el código:

- **FRAME_SIZE:** Tamaño de la ventana de análisis (Default: 4096 muestras). Ventanas más grandes pueden mejorar la compresión en señales estables, pero empeorarla en transitorios rápidos.
- **ORDER:** Orden del filtro LPC (Default: 12). Un orden mayor modela mejor la envolvente espectral pero requiere guardar más coeficientes por trama.

## Requisitos

El proyecto utiliza Python 3 y las siguientes librerías:

```bash
pip install numpy scipy
