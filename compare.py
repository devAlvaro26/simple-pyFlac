import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

PATH_ORIGINAL = "SultansOfSwing.wav"
PATH_ENCODED = "Decoded_Audio.wav"

fs1, audio1 = wavfile.read(PATH_ORIGINAL)
fs2, audio2 = wavfile.read(PATH_ENCODED)

assert len(audio1) == len(audio2), "Audio 1 duration = {}, Audio 2 duration = {}".format(len(audio1), len(audio2))

Equal = True

if audio1.ndim == 2 and audio2.ndim == 2:
    for i in range(len(audio1)):
        if (audio1[i][0] != audio2[i][0]) or (audio1[i][1] != audio2[i][1]):
            Equal = False
            print(f"Signals differ at sample {i}: original={audio1[i]}, decoded={audio2[i]}")
            break
else:
    for i in range(len(audio1)):
        if audio1[i] != audio2[i]:
            Equal = False
            print(f"Signals differ at sample {i}: original={audio1[i]}, decoded={audio2[i]}")
            break

if Equal:
    print("Audios are the same")
else:
    plt.plot(np.linspace(0, len(audio1)/fs1, num=len(audio1)), audio1-audio2, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Difference between Audio Signals')
    plt.show()