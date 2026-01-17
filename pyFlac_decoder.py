import numpy as np
from scipy.io import wavfile
import struct
import concurrent.futures
import os

ENCODED_FILE = "encoded.pyflac"

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
    Decodificación Rice.
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


def mid_side_decode(mid, side, bits_per_sample):
    '''
    Convierte Mid-Side de vuelta a L/R
    r = |Side| % 2 (residuo perdido en división)
    '''
    mid = mid.astype(np.int64)
    side = side.astype(np.int64)
    
    r = np.abs(side) % 2
    
    left = mid + (side + r) // 2
    right = mid + (r - side) // 2
    
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    return left.astype(dtype), right.astype(dtype)


def process_single_frame_decode(args):
    '''
    Función auxiliar para ProcessPoolExecutor.
    '''
    byte_data, k, length, coefs, bits_per_sample = args
    residual = rice_decode(byte_data, k, length).astype(np.int64)
    reconstructed = lpc_synthesis(residual.astype(np.float64), coefs.astype(np.float64))
    
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    samples = np.round(reconstructed).astype(dtype)
    return samples[:length]


def decode_frames(frames_data, use_mid_side, bits_per_sample, leave_one_core=False):
    '''
    Decodifica las tramas procesadas, aplicando síntesis LPC y decodificación Rice.
    Ejecución paralela usando hilos o procesos.
    '''
    num_channels = len(frames_data)
    all_channels = []
    
    for ch in range(num_channels):
        ch_frames = frames_data[ch]
        all_samples = []
        num_frames = len(ch_frames)
        channel_name = "Mid" if (use_mid_side and ch == 0) else "Side" if (use_mid_side and ch == 1) else f"Canal {ch+1}"
        
        print(f"\nDecodificando {num_frames} tramas en paralelo para {channel_name}...")
        
        # Preparar datos para procesamiento paralelo
        frames_to_process = []
        for frame in ch_frames:
            frames_to_process.append((frame['bytes'], frame['k'], frame['length'], frame['coefs'], bits_per_sample))
        
        # Decidir número de workers según CPU
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
        all_channels.append(np.concatenate(all_samples))
    
    if use_mid_side and num_channels == 2:
        print("\nConvirtiendo Mid-Side a L/R...")
        left, right = mid_side_decode(all_channels[0], all_channels[1], bits_per_sample)
        return np.stack([left, right], axis=-1)
    elif num_channels == 1:
        return all_channels[0]
    else:
        # Empaquetar canales en matriz (N, 2)
        min_len = min(len(ch) for ch in all_channels)
        stacked = np.stack([ch[:min_len] for ch in all_channels], axis=-1)
        return stacked


def load_audio_encoded(filename):
    '''
    Carga el archivo binario codificado (soporta mono y estéreo).
    '''
    with open(filename, 'rb') as f:
        # Leer header
        sample_rate = struct.unpack('<I', f.read(4))[0]
        predictor_order = struct.unpack('<H', f.read(2))[0]
        frame_size = struct.unpack('<H', f.read(2))[0]
        num_channels = struct.unpack('<B', f.read(1))[0]
        bits_per_sample = struct.unpack('<B', f.read(1))[0]
        use_mid_side = struct.unpack('<B', f.read(1))[0]
        num_frames = struct.unpack('<I', f.read(4))[0]

        print(f"Sample rate: {sample_rate} Hz")
        print(f"Bits per sample: {bits_per_sample}")
        print(f"Predictor order: {predictor_order}")
        print(f"Frame size: {frame_size}")
        print(f"Canales: {num_channels}")

        # frames_data[channel][frame]
        frames_data = [[] for _ in range(num_channels)]

        for i in range(num_frames):
            for ch in range(num_channels):
                k = struct.unpack('<B', f.read(1))[0]
                padding = struct.unpack('<B', f.read(1))[0]
                length = struct.unpack('<H', f.read(2))[0]
                byte_length = struct.unpack('<H', f.read(2))[0]
                coefs = np.zeros(predictor_order, dtype=np.float32)
                for j in range(predictor_order):
                    coefs[j] = struct.unpack('<f', f.read(4))[0]
                byte_data = f.read(byte_length)
                frames_data[ch].append({
                    'k': k,
                    'padding': padding,
                    'length': length,
                    'bytes': byte_data,
                    'coefs': coefs
                })
            print(f"\rCargando tramas {i+1}/{num_frames}", end='', flush=True)
        print()
        return sample_rate, predictor_order, frame_size, frames_data, num_channels, use_mid_side, bits_per_sample


if __name__ == "__main__":
    try:
        # Cargar archivo codificado
        fs, order, frame_size, frames_data, num_channels, use_mid_side, bits_per_sample = load_audio_encoded(ENCODED_FILE)
        # Decodificar
        audio_data = decode_frames(frames_data, use_mid_side, bits_per_sample)
        # Guardar WAV
        wavfile.write("Decoded_Audio.wav", fs, audio_data)
        print("\nDecodificación completada")
    except FileNotFoundError:
        print(f"Error: Archivo '{ENCODED_FILE}' no encontrado")
    except Exception as e:
        print(f"Error durante la decodificación: {e}")