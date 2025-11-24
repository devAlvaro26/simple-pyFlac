import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

PATH_ORIGINAL = "TunnelOfLove_mono.wav"
PATH_ENCODED = "Decoded_Audio_v2.wav"

fs1, audio1 = wavfile.read(PATH_ORIGINAL)
fs2, audio2 = wavfile.read(PATH_ENCODED)

assert len(audio1) == len(audio2), "Duracion audio 1 = {}, Duracion audio 2 = {}".format(len(audio1), len(audio2))

Igual = True

for i in range(len(audio1)):
    if audio1[i] != audio2[i]:
        Igual = False
        print(f"Las señales difieren en la muestra {i}: original={audio1[i]}, decodificada={audio2[i]}")
        break

if Igual:
    print("Las señales son exactamente iguales")
else:
    plt.plot(np.linspace(0, len(audio1)/fs1, num=len(audio1)), audio1-audio2, alpha=0.7)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Diferencia entre Señales de Audio')
    plt.show()