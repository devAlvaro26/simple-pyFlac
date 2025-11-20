# Compresor de Audio Lossless (Predictor Lineal + Huffman)

Este repositorio contiene una implementación en Python de un **códec de audio sin pérdidas (lossless)**. El sistema utiliza técnicas de predicción lineal para decorrelar la señal y codificación de Huffman para comprimir la entropía del residuo resultante.

## 📋 Descripción del Proyecto

El objetivo es reducir el tamaño de archivos de audio `.wav` (mono, 16-bit PCM) sin perder información al reconstruirlos. El flujo de procesamiento es el siguiente:

1.  **Lectura:** Se lee el audio RAW.
2.  **Predicción Lineal:** Se estima la muestra actual $x[n]$ basándose en muestras anteriores ($x[n-1], x[n-2]$).
3.  **Cálculo del Residuo:** Se obtiene la diferencia entre la señal real y la predicción ($e[n] = x[n] - \hat{x}[n]$). El residuo tiene una varianza mucho menor que la señal original, reduciendo su entropía.
4.  **Codificación Huffman:** Se asignan códigos de longitud variable a los valores del residuo según su frecuencia de aparición.
5.  **Empaquetado:** Se guarda el bitstream y la tabla de códigos en un archivo binario (`.bin`).

### Modelos de Predicción Soportados
El sistema soporta predictores de orden 1 y 2:
- **Orden 1:** $\hat{x}[n] = x[n-1]$
- **Orden 2:** $\hat{x}[n] = 2x[n-1] - x[n-2]$

## 📂 Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `codec.py` | **Script principal**. Ejecuta el ciclo completo: carga audio, comprime, guarda, descomprime y compara la señal reconstruida con la original. |
| `encoder.py` | Módulo encargado de la lectura del WAV, cálculo del residuo y generación del bitstream Huffman. |
| `decoder.py` | Módulo que lee el archivo binario, decodifica el bitstream y reconstruye el audio a partir del residuo. |
| `encoded.bin` | Ejemplo de archivo de salida comprimido (generado por el encoder). |

## 🛠️ Requisitos

El proyecto utiliza Python 3 y las siguientes librerías científicas:

```bash
pip install numpy scipy huffman
