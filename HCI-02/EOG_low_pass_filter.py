import numpy as np
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt

EOG_signal = open('EOG.txt', 'r')
lines = EOG_signal.readlines()
print(lines)
Amp = []
for i in range(len(lines) - 1):
    L = lines[i + 1]
    Amp.append(int(L))

print(Amp)
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, len(Amp)), Amp)
plt.xlabel('Time')
plt.ylabel('Amp')


def butter_lowpass_filter(data, cutoff, SamplingRate, order):
    nyq = 0.5 * SamplingRate
    normal_cutoff = cutoff / nyq
    # get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False, fs=None)
    Filtered_Data = filtfilt(b, a, data)
    return Filtered_Data


Filtered_signal = butter_lowpass_filter(Amp, cutoff=2, SamplingRate=176, order=2)
print(Filtered_signal)
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, len(Filtered_signal)), Filtered_signal)
plt.xlabel("Time(S)")
plt.ylabel("Amplitude(V)")
plt.show()
