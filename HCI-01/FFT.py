import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
import math

plt.style.use('seaborn-v0_8-poster')

# sampling rate
sr = 2000
# sampling interval
ts = 1.0 / sr
t = np.arange(0, 1, ts)

freqs = [1, 4, 7]
x = np.zeros_like(t)

for freq in freqs:
    x += np.sin(2 * np.pi * freq * t)

X = fft(x)

phase = [math.atan2(X.imag[i], X.real[i]) for i in range(len(X.imag))]
degrees = [p * (180 / math.pi) for p in phase]

plt.figure(figsize=(12, 6))
plt.subplot(121)

plt.stem(freqs, degrees[:len(freqs)], 'b', markerfmt=" ", basefmt="-b")

plt.xlabel('Freq (Hz)')
plt.ylabel('FFT phase shifts')
plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, ifft(x), 'r')
plt.xlabel('Time (S)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
