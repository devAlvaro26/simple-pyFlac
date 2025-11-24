import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import pickle
import huffman

PATH_AUDIO_ENCODED = "encoded.bin"

def read_audio_encoded(filename):
    '''
    Cargar audio comprimido desde archivo
    '''
    with open(filename, 'rb') as f:
        compressed_data = pickle.load(f)
    
    print(f"\nArchivo comprimido cargado: '{filename}'")
    
    byte_data = compressed_data['byte_data']
    padding = compressed_data['padding']
    codes = compressed_data['huffman_codes']
    predictor_order = compressed_data['predictor_order']
    fs = compressed_data['sample_rate']
    
    # Convertir bytes a bitstream
    bitstream = ''.join(format(byte, '08b') for byte in byte_data)
    # Remover padding
    if padding > 0:
        bitstream = bitstream[:-padding]
        
    return bitstream, codes, fs, predictor_order
    

def huffman_decode(bitstream, codes):
    '''
    Decodificación Huffman
    '''
    inv_codes = {v: k for k, v in codes.items()}
    decoded = []
    buffer = ""
    
    for bit in bitstream:
        buffer += bit
        if buffer in inv_codes:
            decoded.append(inv_codes[buffer])
            buffer = ""
    
    return np.array(decoded, dtype=np.int16)


def reconstruct_audio(residual, order=3):
    '''
    Reconstruir audio original desde el residual
    '''
    reconstructed = np.zeros_like(residual)
    
    if order == 1:
        reconstructed[0] = residual[0]
        for i in range(1, len(residual)):
            reconstructed[i] = residual[i] + reconstructed[i-1]
            
    elif order == 2:
        reconstructed[0] = residual[0]
        reconstructed[1] = residual[1]
        for i in range(2, len(residual)):
            reconstructed[i] = residual[i] + 2*reconstructed[i-1] - reconstructed[i-2]
    elif order == 3:
        reconstructed[0] = residual[0]
        reconstructed[1] = residual[1]
        reconstructed[2] = residual[2]
        for i in range(3, len(residual)):
            reconstructed[i] = residual[i] + 3*reconstructed[i-1] - 3*reconstructed[i-2] + reconstructed[i-3]
    else:
        raise ValueError("Solo se soporta orden 1, 2 o 3")
    
    return reconstructed

if __name__ == "__main__":
    audio_encoded, codes, fs, order = read_audio_encoded(PATH_AUDIO_ENCODED)

    residual = huffman_decode(audio_encoded,codes)

    audio = reconstruct_audio(residual, order)

    write("Decoded_Audio.wav",fs,audio)