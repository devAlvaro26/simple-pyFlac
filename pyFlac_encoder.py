import numpy as np
from scipy.io import wavfile
import struct
import concurrent.futures
import os

PATH_AUDIO = "SultansOfSwing.wav"
FRAME_SIZE = 4096   # Tamaño de trama en muestras
ORDER = 12          # Orden del predictor LPC (4-16)

def read_file(path):
    '''
    Lee un archivo de audio y lo devuelve como un array de muestras.
    Soporta mono y estéreo.
    '''
    try:
        fs, data = wavfile.read(path)
        if data.dtype != np.int16 and data.dtype != np.int32:
            raise ValueError("Solo se soporta audio int16 o int32")

        print(f"Frecuencia de muestreo: {fs} Hz")
        print(f"Tipo de datos: {data.dtype}")
        print(f"Tamaño: {data.shape}")

        if data.ndim == 1:
            # Mono
            return fs, 1, [data]
        elif data.ndim == 2 and data.shape[1] == 2:
            # Estéreo
            return fs, 2, [data[:, 0], data[:, 1]]
        else:
            raise ValueError("Solo se soporta audio mono o estéreo (2 canales)")

    except FileNotFoundError:
        print(f"Error: Archivo '{path}' no encontrado")
        return None, None, None
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return None, None, None


def mid_side_encode(left, right):
    '''
    Convierte canales L/R a Mid-Side.
    '''

    left = left.astype(np.int64)
    right = right.astype(np.int64)
    
    mid = (left + right) // 2
    side = left - right

    return mid, side


def autocorr(x, order):
    '''
    Autocorrelación de orden dado.
    '''
    N = len(x)
    r = np.zeros(order + 1, dtype=np.float64)
    full = np.correlate(x, x, mode='full')
    mid = len(full)//2
    for k in range(order+1):
        r[k] = full[mid + k]
    return r


def levinson_durbin(r, order):
    '''
    Algoritmo Levinson-Durbin.
    '''
    a = np.zeros(order+1, dtype=np.float64)
    a[0] = 1.0
    e = r[0]
    if e == 0:
        return np.zeros(order), 0.0

    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc -= a[j] * r[i-j]
        k = acc / e
        a_prev = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_prev[j] - k * a_prev[i-j]
        e = e * (1.0 - k*k)
        if e <= 0:
            e = 1e-12

    return a[1:], e


def LPC(frame, order):
    '''
    Calcula coeficientes LPC y valor residual para una trama.
    '''
    frame = frame.astype(np.float64)
    r = autocorr(frame, order)
    coefs, e = levinson_durbin(r, order)
    coefs = coefs.astype(np.float32)
    N = len(frame)

    predicted = np.zeros(N, dtype=np.float64)
    residual = np.zeros(N, dtype=np.int64)
    
    # Las primeras 'order' muestras se guardan crudas
    if N > 0:
        head = min(order, N)
        residual[:head] = np.int64(np.round(frame[:head]))

    # Vectorizar la predicción usando convolución
    if N > order:
        conv = np.convolve(frame, coefs, mode='full')
        # Para n en [order, N-1], conv[n-1]
        predicted_section = conv[order-1:N-1]
        predicted[order:N] = predicted_section

        # Redondear y calcular residuo de forma vectorizada
        predicted_int = np.round(predicted[order:N]).astype(np.int64)
        residual[order:N] = np.int64(np.round(frame[order:N])) - predicted_int

    return residual, coefs, e


def zigzag_encode(x):
    '''
    Zigzag encoding para enteros.
    '''
    x = int(x)
    if x >= 0:
        return 2 * x
    else:
        return -2 * x - 1


def optimal_k(residual):
    '''
    Estima el mejor valor de k.
    '''
    if len(residual) == 0:
        return 0

    # Vectorizar zigzag y cálculo de bits por k usando numpy
    res = np.asarray(residual, dtype=np.int64)
    folded = np.where(res >= 0, res * 2, -res * 2 - 1).astype(np.int64)

    min_bits = np.inf
    k = 0
    for k_opt in range(33):
        q = folded >> k_opt
        # bits por muestra: q (ceros) + 1 (uno) + k_opt (remainder)
        total_bits = int(np.sum(q + 1 + k_opt))
        if total_bits < min_bits:
            min_bits = total_bits
            k = k_opt

    return k


def rice_encode(residual, k):
    '''
    Codificación Rice
    '''
    bits = []
    
    for sample in residual:
        # Zigzag encoding para enteros
        folded = zigzag_encode(sample)
        
        # Dividir en quotient y remainder
        q = folded >> k
        r = folded & ((1 << k) - 1)
        
        # Unary para quotient: q ceros seguidos de un uno
        for _ in range(q):
            bits.append(0)
        bits.append(1)
        
        # Binario para remainder (k bits, MSB primero)
        for i in range(k - 1, -1, -1):
            bits.append((r >> i) & 1)
    
    return bits


def bits_to_bytes(bits):
    '''
    Convierte lista de bits a bytes.
    '''
    padding = (8 - (len(bits) % 8)) % 8
    if padding:
        bits.extend([0] * padding)
    
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i + j]
        byte_array.append(byte_val)
    
    return bytes(byte_array), padding


def process_single_frame(args):
    '''
    Función auxiliar para ProcessPoolExecutor.
    '''
    frame, valid_length, order = args
    residual, coefs, error = LPC(frame, order)
    residual = residual[:valid_length]
    k = optimal_k(residual)
    bits = rice_encode(residual, k)
    byte_data, padding = bits_to_bytes(bits)
    return {
        'bytes': byte_data,
        'padding': padding,
        'k': k,
        'coefs': coefs,
        'length': valid_length,
    }


def process_frames(data, order, frame_size=FRAME_SIZE, leave_one_core=False):
    '''
    Procesa el audio en tramas, aplicando LPC y codificación Rice.
    Ejecucución paralela usando hilos o procesos.
    '''
    num_samples = len(data)
    num_frames = (num_samples + frame_size - 1) // frame_size
    
    frames_data = []
    
    # Preparar datos para procesamiento paralelo: (frame_padded, valid_length)
    frames_to_process = []
    for i in range(num_frames):
        start = i * frame_size
        end = min(start + frame_size, num_samples)
        frame = data[start:end]
        valid_length = len(frame)

        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - valid_length), mode='constant')

        frames_to_process.append((frame, valid_length, order))

    print(f"\nProcesando {num_frames} tramas de {frame_size} muestras en paralelo")

    # Decidir número de workers según CPU
    max_workers = os.cpu_count() - 1 if leave_one_core else os.cpu_count()
    max_workers = min(max_workers, num_frames)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, result in enumerate(executor.map(process_single_frame, frames_to_process), start=1):
                frames_data.append(result)
                print(f"\rTrama {i}/{num_frames} procesada", end='', flush=True)
    except Exception as e:
        # Fallback a hilos si falla la ejecución por procesos
        print(f"Error al ejecutar con procesos, usando hilos en su lugar. Error: {e}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, result in enumerate(executor.map(process_single_frame, frames_to_process), start=1):
                frames_data.append(result)
                print(f"\rTrama {i}/{num_frames} procesada", end='', flush=True)
    
    print("\n")

    return frames_data


def save_audio_encoded(filename, frames_data, sample_rate, predictor_order, frame_size, use_mid_side, bits_per_sample):
    '''
    Guarda en formato binario. frames_data debe ser una lista de listas (por canal).
    '''
    num_channels = len(frames_data)
    num_frames = len(frames_data[0])
    with open(filename, 'wb') as f:
        # Cabeceras de datos
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<H', predictor_order))
        f.write(struct.pack('<H', frame_size))
        f.write(struct.pack('<B', num_channels))
        f.write(struct.pack('<B', bits_per_sample))
        f.write(struct.pack('<B', 1 if use_mid_side else 0))
        f.write(struct.pack('<I', num_frames))
        # Para cada trama y canal: metadata + coeficientes + datos
        for frame_idx in range(num_frames):
            for ch in range(num_channels):
                frame = frames_data[ch][frame_idx]
                f.write(struct.pack('<B', frame['k']))
                f.write(struct.pack('<B', frame['padding']))
                f.write(struct.pack('<H', frame['length']))
                f.write(struct.pack('<H', len(frame['bytes'])))
                # Coeficientes
                for coef in frame['coefs']:
                    f.write(struct.pack('<f', coef))
                # Datos comprimidos
                f.write(frame['bytes'])
    # Estadísticas
    compressed_size = sum(sum(len(frame['bytes']) for frame in ch_frames) for ch_frames in frames_data)
    original_size = sum(sum(frame['length'] for frame in ch_frames) for ch_frames in frames_data) * 2
    ratio = (compressed_size / original_size) * 100 if original_size > 0 else 0.0
    print("================= Estadísticas de Compresión ================")
    print(f"Tamaño original: {original_size:,} bytes")
    print(f"Tamaño comprimido: {compressed_size:,} bytes")
    print(f"Ratio: {ratio:.2f}%")


if __name__ == "__main__":
    try:
        fs, num_channels, data_channels = read_file(PATH_AUDIO)
        if data_channels is not None:
            # Detectar bits por muestra
            bits_per_sample = 0
            if data_channels[0].dtype == np.int16:
                bits_per_sample = 16
            elif data_channels[0].dtype == np.int32:
                bits_per_sample = 32
            
            frames_data = []
            
            # Mid-Side encoding
            if num_channels == 2:
                mid, side = mid_side_encode(data_channels[0], data_channels[1])
                data_to_process = [mid, side]
                use_mid_side = True
                channel_names = ["Mid", "Side"]
            else:
                data_to_process = data_channels
                use_mid_side = False
                channel_names = [f"Canal {i+1}" for i in range(num_channels)]
            
            for ch, data in enumerate(data_to_process):
                print(f"\nProcesando {channel_names[ch]} ({ch+1}/{num_channels})")
                frames = process_frames(data, ORDER, frame_size=FRAME_SIZE)
                frames_data.append(frames)
            
            save_audio_encoded("encoded.pyflac", frames_data, fs, ORDER, FRAME_SIZE, use_mid_side, bits_per_sample)
        else:
            print("El archivo está vacío")
    except FileNotFoundError:
        print(f"Error: Archivo '{PATH_AUDIO}' no encontrado")
    except Exception as e:
        print(f"Error durante la codificación: {e}")