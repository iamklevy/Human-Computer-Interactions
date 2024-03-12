import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels.api as sm

with open('p1.txt') as f:
    lines = f.readlines()

sub1 = []

for i in range(len(lines) - 1):
    L = lines[i + 1].split()
    sub1.append(int(L[0]))

with open('p38.txt') as f:
    lines = f.readlines()

sub2 = []

for i in range(len(lines) - 1):
    L = lines[i + 1].split()
    sub2.append(int(L[0]))

seg1 = np.array(sub1[0:1000])
seg2 = np.array(sub2[0:1000])

plt.figure(figsize=(24, 6))
plt.plot(np.arange(0, len(seg1)), seg1)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(s)")
plt.show()

plt.plot(np.arange(0, len(seg2)), seg2)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(s)")
plt.show()


Acc = sm.tsa.acf(seg1, len(seg1))
seg3 = np.array(Acc[0:100])

plt.plot(np.arange(0, len(Acc)), Acc, "o")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(s)")
plt.show()

DCT = scipy.fftpack.dct(seg3, 2)

plt.plot(np.arange(0, len(DCT)), DCT, "r")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(s)")
plt.show()