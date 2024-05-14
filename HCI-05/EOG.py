import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, pi, arange
from scipy.fftpack import fft
import scipy

sr = 1000
nsample = 200
t = arange(nsample) / sr
x = cos(2 * pi * 100 * t) + cos(2 * pi * 70 * t)

N0 = -10  # Noise level (dB)
x += np.random.randn(len(x)) * 10 ** (N0 / 20.0)
plt.figure(figsize=(12, 6))
plt.plot((np.arange(0, len(x))), x, 'r')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

X = fft(x)
N = len(X)
n = np.arange(N)
T = N / sr
freq = n / T

plt.figure(figsize=(12, 6))
plt.stem(freq, np.abs(X), 'b', markerfmt="", basefmt="-b")
plt.xlim(0, 500)
plt.xlabel('Freq(HZ)')
plt.ylabel('FFT Amplitude')

plt.show()

(F, S) = scipy.signal.periodogram(X, sr, scaling='density')

plt.semilogy(F, S)
plt.ylim([1e-7, 1e2])


plt.show()