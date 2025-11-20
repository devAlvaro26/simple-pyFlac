import numpy as np
from scipy.io import wavfile
import pickle
import huffman

PATH_AUDIO = "SultansOfSwingMono.wav"

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


def reconstruct_audio(residual, order=2):
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
    
    return reconstructed


def compression_stats(original, residual, bitstream):
    '''
    Calcular y mostrar estadísticas de compresión
    '''
    original_bits = len(original) * 16  # int16 = 16 bits por muestra
    compressed_bits = len(bitstream)
    ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
    
    print("\n" + "="*50)
    print("ESTADÍSTICAS DE COMPRESIÓN")
    print("="*50)
    print(f"Tamaño original:    {original_bits:,} bits ({original_bits/8:,.0f} bytes)")
    print(f"Tamaño comprimido:  {compressed_bits:,} bits ({compressed_bits/8:,.0f} bytes)")
    print(f"Ratio de compresión: {ratio:.2f}:1")
    print(f"Ahorro de espacio:   {(1 - 1/ratio)*100:.1f}%")
    print("="*50)
    
    # Estadísticas del residual
    print(f"\nEstadísticas del residual:")
    print(f"  Media: {np.mean(residual):.2f}")
    print(f"  Desv. estándar: {np.std(residual):.2f}")
    print(f"  Rango: [{np.min(residual)}, {np.max(residual)}]")
    print(f"  Valores únicos: {len(np.unique(residual))}")


# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    print("COMPRESOR DE AUDIO CON PREDICTOR Y HUFFMAN")
    print("="*50 + "\n")
    
    # Leer archivo
    fs, audio = readFile(PATH_AUDIO)
    
    if audio is None:
        print("No se pudo procesar el archivo")
    
    else:
        # Usar subset para pruebas
        audio_segment = audio
        print(f"\nProcesando segmento de {len(audio_segment)} muestras\n")
        
        # COMPRESIÓN
        print("PASO 1: Predicción")
        predictor_order = 2
        residual, predicted = predictor(audio_segment, order=predictor_order)
        print(f"Predictor de orden {predictor_order} aplicado")
        
        print("\nPASO 2: Codificación Huffman")
        bitstream, codes = huffman_encode(residual)
        print(f"Códigos Huffman generados: {len(codes)} símbolos únicos")
        
        # Estadísticas
        compression_stats(audio_segment, residual, bitstream)

        # Guardar Archivo
        save_audio_encoded("encoded.bin",bitstream,codes,fs,predictor_order)
        
        # DESCOMPRESIÓN
        print("\n" + "="*50)
        print("PROCESO DE DESCOMPRESIÓN")
        print("="*50)
        
        print("\nPASO 3: Decodificación Huffman")
        decoded_residual = huffman_decode(bitstream, codes)
        print(f"Residual decodificado: {len(decoded_residual)} muestras")
        
        print("\nPASO 4: Reconstrucción de audio")
        reconstructed_audio = reconstruct_audio(decoded_residual, order=predictor_order)
        print(f"Audio reconstruido: {len(reconstructed_audio)} muestras")
        
        # Comparación de primeras muestras
        print("\nComparación de 10 muestras:")
        print("Original:       ", audio_segment[9990:10000])
        print("Reconstruido:   ", reconstructed_audio[9990:10000])
        print("Diferencia:     ", audio_segment[9990:10000] - reconstructed_audio[9990:10000])
        
        # Guardar audio reconstruido (opcional)
        # wavfile.write("reconstructed.wav", fs, reconstructed_audio)