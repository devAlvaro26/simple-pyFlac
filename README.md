# Compresor de Audio Lossless (LPC + Rice Coding)

Implementación en Python de un **códec de audio sin pérdidas (lossless)**. El sistema utiliza **Codificación Predictiva Lineal (LPC)** para modelar la señal espectralmente y **Codificación Rice** (Golomb-Rice) con adaptación dinámica de parámetros para comprimir el residuo.

## 📋 Descripción del Proyecto

Comprimir una señal PCM sin pérdidas estilo FLAC.

### Flujo de Procesamiento (Encoder)

1.  **Tramado (Framing):** Segmentación entramas (por defecto 4096 muestras).
2.  **Análisis LPC (Levinson-Durbin):** Para cada trama, se calcula la autocorrelación y se utiliza el algoritmo de Levinson-Durbin para encontrar los coeficientes óptimos del filtro predictor (orden configurable, por defecto 12).
3.  **Cálculo del Residuo:** Se predice la señal actual $\hat{x}[n]$ mediante la combinación lineal de muestras pasadas y los coeficientes LPC. La diferencia con la señal real es el residuo:
    $$e[n] = x[n] - \text{round}(\hat{x}[n])$$
   .
4.  **Codificación de Entropía (Rice):**
    - **Zigzag Encoding:** Convierte el residuo (con signo) a enteros positivos para optimizar la codificación ($0 \to 0, -1 \to 1, 1 \to 2...$).
    - **Estimación de K:** Se calcula el parámetro $k$ óptimo para la codificación Rice basándose en la media absoluta del residuo de la trama actual.
    - **Rice Coding:** Se genera el bitstream comprimido separando el valor en cociente (unario) y resto (binario).
5.  **Empaquetado:** Se guarda un archivo binario (`.bin`) que contiene las cabeceras globales, y para cada trama: su metadata ($k$, padding, longitud), los coeficientes LPC y el bitstream comprimido.

### Flujo de Decodificación (Decoder)

1.  **Lectura de Tramas:** Se extraen los parámetros $k$ y los coeficientes LPC de cada bloque.
2.  **Decodificación Rice y Zigzag:** Se recupera el residuo original $e[n]$.
3.  **Síntesis LPC:** Se reconstruye la señal sumando el residuo a la predicción generada por los coeficientes recuperados:
    $$x[n] = e[n] + \text{round}(\hat{x}[n])$$
   .

## 📂 Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `encoder_v2.py` | Script de codificación. Lee el WAV, aplica LPC (Levinson-Durbin), estima el parámetro $k$ de Rice y genera el archivo binario `encoded_v2.bin`. |
| `decoder_v2.py` | Script de decodificación. Lee el binario, reconstruye el audio mediante síntesis LPC y guarda el archivo `Decoded_Audio_v2.wav`. |
| `encoded_v2.bin` | Archivo de salida comprimido generado por el encoder. |
| `SultansOfSwing_mono.wav` | Archivo de audio de ejemplo (entrada del encoder). |

## ⚙️ Configuración del Algoritmo

El sistema permite ajustar los siguientes parámetros en el código:

- **FRAME_SIZE:** Tamaño de la ventana de análisis (Default: 4096 muestras). Ventanas más grandes pueden mejorar la compresión en señales estables, pero empeorarla en transitorios rápidos.
- **Predictor Order:** Orden del filtro LPC (Default: 12). Un orden mayor modela mejor la envolvente espectral pero requiere guardar más coeficientes por trama.

## 🛠️ Requisitos

El proyecto utiliza Python 3 y las siguientes librerías estándar científicas:

```bash
pip install numpy scipy
