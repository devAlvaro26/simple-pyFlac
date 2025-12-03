import numpy as np
from scipy.io import wavfile
import struct
import concurrent.futures
import os

ENCODED_FILE = "encoded_v3.bin"

def bytes_to_bits(byte_data):
    '''
    Convierte bytes a lista de bits.
    '''
    bits = []
    for byte in byte_data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def zigzag_decode(x):
    '''
    Decodificación zigzag.
    '''
    x = int(x)
    if x % 2 == 0:
        return x // 2
    else:
        return -(x + 1) // 2


def rice_decode(byte_data, k, num_samples):
    '''
    Decodificacion Rice.
    '''
    decoded = []

    bits = bytes_to_bits(byte_data)
    i = 0
    
    while len(decoded) < num_samples and i < len(bits):
        # Leer unary: contar ceros hasta encontrar un uno
        q = 0
        while i < len(bits) and bits[i] == 0:
            q += 1
            i += 1
        
        if i >= len(bits):
            break
        
        # Saltar el uno del unary
        i += 1
        
        # Leer k bits para el resto
        if i + k > len(bits):
            break
        
        r = 0
        for j in range(k):
            r = (r << 1) | bits[i]
            i += 1
        
        # Reconstruir folded value
        folded = (q << k) | r
        
        # Aplicar zigzag decoding
        sample = zigzag_decode(folded)
        decoded.append(sample)
    
    return np.array(decoded, dtype=np.int64)


def process_single_frame_decode(args):
    '''
    Función para decodificar una sola trama (para uso en pools).
    args: (byte_data, k, length, coefs)
    '''
    byte_data, k, length, coefs = args
    residual = rice_decode(byte_data, k, length).astype(np.int64)
    reconstructed = lpc_synthesis(residual.astype(np.float64), coefs.astype(np.float64))
    samples = np.round(reconstructed).astype(np.int16)
    return samples[:length]


def lpc_synthesis(residual, coefs):
    '''
    Reconstruye la señal a partir del residual y coeficientes LPC.
    '''
    order = len(coefs)
    N = len(residual)
    reconstructed = np.zeros(N, dtype=np.float64)
    
    # Copiar las primeras 'order' muestras
    if order > 0:
        reconstructed[:order] = residual[:order].astype(np.float64)
    
    for n in range(order, N):
        prediction = np.dot(coefs, reconstructed[n-order:n][::-1])
        
        if not np.isfinite(prediction):
            prediction = 0.0
        
        # Redondear la predicción antes de sumar el residuo
        prediction = np.round(prediction)

        # Reconstruir la muestra
        reconstructed[n] = residual[n] + prediction
    
    return reconstructed


def load_audio_encoded(filename):
    '''
    Carga el archivo binario codificado.
    '''
    with open(filename, 'rb') as f:
        # Leer header
        sample_rate = struct.unpack('<I', f.read(4))[0]
        predictor_order = struct.unpack('<H', f.read(2))[0]
        frame_size = struct.unpack('<H', f.read(2))[0]
        num_frames = struct.unpack('<I', f.read(4))[0]
        
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Predictor order: {predictor_order}")
        print(f"Frame size: {frame_size}")
        print(f"Número de tramas: {num_frames}")
        
        frames_data = []
        
        for i in range(num_frames):
            # Leer metadata de la trama
            k = struct.unpack('<B', f.read(1))[0]
            padding = struct.unpack('<B', f.read(1))[0]
            length = struct.unpack('<H', f.read(2))[0]
            byte_length = struct.unpack('<H', f.read(2))[0]
            
            # Leer coeficientes
            coefs = np.zeros(predictor_order, dtype=np.float32)
            for j in range(predictor_order):
                coefs[j] = struct.unpack('<f', f.read(4))[0]
            
            # Leer datos comprimidos
            byte_data = f.read(byte_length)
            
            frames_data.append({
                'k': k,
                'padding': padding,
                'length': length,
                'bytes': byte_data,
                'coefs': coefs
            })
            
            print(f"\rCargando trama {i+1}/{num_frames}", end='', flush=True)
        
        print()
        return sample_rate, predictor_order, frame_size, frames_data


def decode_frames(frames_data, leave_one_core=True):
    '''
    Decodificar todas las tramas y reconstruir el audio completo.
    '''
    all_samples = []
    num_frames = len(frames_data)

    print(f"\nDecodificando {num_frames} tramas en paralelo...")

    # Preparar argumentos para cada trama
    frames_to_process = []
    for frame in frames_data:
        frames_to_process.append((frame['bytes'], frame['k'], frame['length'], frame['coefs']))

    # Elegir número de workers
    max_workers = os.cpu_count() - 1 if leave_one_core else os.cpu_count()
    max_workers = min(max_workers, num_frames)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, samples in enumerate(executor.map(process_single_frame_decode, frames_to_process), start=1):
                all_samples.append(samples)
                print(f"\rTrama {i}/{num_frames} decodificada", end='', flush=True)
    except Exception as e:
        # Fallback a hilos si falla la ejecución por procesos
        print(f"Error al ejecutar con procesos, usando hilos en su lugar. Error: {e}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, samples in enumerate(executor.map(process_single_frame_decode, frames_to_process), start=1):
                all_samples.append(samples)
                print(f"\rTrama {i}/{num_frames} decodificada", end='', flush=True)

    print()
    return np.concatenate(all_samples)

if __name__ == "__main__":
    try:
        # Cargar archivo codificado
        fs, order, frame_size, frames_data = load_audio_encoded(ENCODED_FILE)
        
        # Decodificar
        audio_data = decode_frames(frames_data)
        
        # Guardar WAV
        wavfile.write("Decoded_Audio_v3.wav", fs, audio_data)
        
        print("\nDecodificación completada")
        
    except FileNotFoundError:
        print(f"Error: Archivo '{ENCODED_FILE}' no encontrado")
    except Exception as e:
        print(f"Error durante la decodificación: {e}")