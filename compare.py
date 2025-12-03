import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

PATH_ORIGINAL = "SultansOfSwing_mono.wav"
PATH_ENCODED = "Decoded_Audio_v3.wav"

fs1, audio1 = wavfile.read(PATH_ORIGINAL)
fs2, audio2 = wavfile.read(PATH_ENCODED)

assert len(audio1) == len(audio2), "Duracion audio 1 = {}, Duracion audio 2 = {}".format(len(audio1), len(audio2))

Igual = True

if audio1.ndim == 2 and audio2.ndim == 2:
    for i in range(len(audio1)):
        if (audio1[i][0] != audio2[i][0]) or (audio1[i][1] != audio2[i][1]):
            Igual = False
            print(f"Las señales difieren en la muestra {i}: original={audio1[i]}, decodificada={audio2[i]}")
            break
else:
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