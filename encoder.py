import numpy as np
from scipy.io import wavfile
import pickle
import huffman

PATH_AUDIO = "SultansOfSwing_mono.wav"

def readFile(path):
    '''
    Leer archivos wav int16 mono con validaciones
    '''
    try:
        fs, data = wavfile.read(path)
        
        # Validaciones
        if data.ndim != 1:
            raise ValueError("Solo se soporta audio mono")
        if data.dtype != np.int16:
            raise ValueError("Solo se soporta audio int16")
        
        print(f"Frecuencia de muestreo: {fs} Hz")
        print(f"Tipo de datos: {data.dtype}")
        print(f"Tamaño: {data.shape}")
        print(f"Duración: {len(data)/fs:.2f} segundos")
        
        return fs, data
        
    except FileNotFoundError:
        print(f"Error: Archivo '{path}' no encontrado")
        return None, None
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return None, None


def predictor(x, order=2):
    '''
    Predictor lineal de orden N
    order=1: x[n] = x[n-1]
    order=2: x[n] = 2*x[n-1] - x[n-2]
    '''
    predicted = np.zeros_like(x)
    
    if order == 1:
        predicted[1:] = x[:-1]
    elif order == 2:
        predicted[1] = x[0]
        predicted[2:] = 2*x[1:-1] - x[:-2]
    else:
        raise ValueError("Solo se soporta orden 1 o 2")
    
    residual = x - predicted
    return residual, predicted


def huffman_encode(residual):
    '''
    Codificación Huffman del residual
    '''
    # Convertir a lista
    residual_list = residual.tolist()
    
    # Calcular frecuencias de cada símbolo
    unique, counts = np.unique(residual_list, return_counts=True)
    freq_dict = dict(zip(unique.tolist(), counts.tolist()))
    
    # Generar códigos Huffman desde el diccionario de frecuencias
    codes = huffman.codebook(freq_dict.items())
    
    # Generar bitstream
    bitstream = "".join(codes[sym] for sym in residual_list)
    
    return bitstream, codes

def save_audio_encoded(filename, bitstream, codes, sample_rate, predictor_order):
    '''
    Guardar audio comprimido en archivo binario
    '''
    # Calcular padding necesario
    padding = (8 - len(bitstream) % 8) % 8
    bitstream_padded = bitstream + '0' * padding
    
    # Convertir a bytes
    num_bytes = len(bitstream_padded) // 8
    byte_array = bytearray()
    
    for i in range(num_bytes):
        byte = bitstream_padded[i*8:(i+1)*8]
        byte_array.append(int(byte, 2))
    
    byte_data = bytes(byte_array)

    compressed_data = {
        'byte_data': byte_data,
        'huffman_codes': codes,
        'padding' : padding,
        'sample_rate': sample_rate,
        'predictor_order': predictor_order
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)
    
    file_size = len(byte_data)
    print(f"\nArchivo comprimido guardado: '{filename}'")
    print(f"Tamaño: {file_size:,} bytes ({file_size/1024:.2f} KB)")


if __name__ == "__main__":
    order = 2

    fs, audio = readFile(PATH_AUDIO)

    residual, predicted = predictor(audio,order)

    bitstream, codes = huffman_encode(residual)

    save_audio_encoded("encoded.bin",bitstream, codes, fs, order)