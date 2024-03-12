import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks

with open('ECG.txt') as f:
    lines = f.readlines()

ecgX = []
ecgY = []
for i in range(len(lines) - 1):
    L = lines[i + 1].split()
    ecgX.append(int(L[0]))
    ecgY.append(int(L[1]))
print(ecgY)


def butter_Bandpass_filter(data, Low_Cutoff, High_Cutoff, SamplingRate, order):
    nyq = 0.5 * SamplingRate
    low = Low_Cutoff / nyq
    high = High_Cutoff / nyq
    b, a = butter(order, [low, high], btype='band', analog=False, fs=None)
    Filtered_Data = filtfilt(b, a, data)
    return Filtered_Data


Filtered_signal = butter_Bandpass_filter(ecgY, Low_Cutoff=1, High_Cutoff=40, SamplingRate=250, order=2)
print(Filtered_signal)

arr_x = np.array(ecgX)
arr_y = np.array(Filtered_signal)
x = np.diff(arr_x)
y = np.diff(arr_y)
dy = np.zeros(400)
dy[0:len(dy) - 1] = y / x
print("First order difference  : ", dy)

plt.figure(figsize=(24, 6))
plt.subplot(151)
plt.plot(ecgX, ecgY)

plt.subplot(152)
plt.plot(np.arange(0, len(Filtered_signal)), Filtered_signal)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(v)")

plt.subplot(153)
plt.plot(np.arange(0, len(dy)), dy)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(v)")

result = [dy[i] ** 2 for i in range(len(dy))]
print(result)
plt.subplot(154)
plt.plot(np.arange(0, len(result)), result)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(v)")

win_size = round(0.03 * 250);
sum = 0

for j in range(win_size):
    sum += result[j] / win_size
    result[j] = sum

# Apply the moving window integration using the equation given
for index in range(win_size, len(result)):
    sum += result[index] / win_size
    sum -= result[index - win_size] / win_size
    result[index] = sum

print(result)
plt.subplot(155)
plt.plot(np.arange(0, len(result)), result)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude(v)")

peaks, _ = find_peaks(result)
X = []
Y = []
for i in range(len(peaks) - 1):
    L = (peaks[i])
    Y.append(result[L])
    X.append(peaks[i])
plt.plot(X, Y, "x")

R_peaks = []
X_index = []

for i in range(len(peaks) - 1):
    L = (peaks[i])
    if result[L] > 60000:
        R_peaks.append(result[L])
        X_index.append(L)
plt.plot(X_index, R_peaks, "o")
plt.show()

ls = []
lsx = []

for i in range(len(peaks) - 1):
    L = (peaks[i])
    if 30000 < result[L] < 60000:
        ls.append(result[L])
        lsx.append(L)
plt.scatter(np.array(lsx), np.array(ls))
plt.show()
print(ls)
