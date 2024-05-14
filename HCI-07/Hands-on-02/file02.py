import numpy as np
import pandas as pd
import scipy
import scipy.fftpack as fft
import statsmodels.api as sm

ecg_data = pd.read_excel("AC_DCT dataset.xlsx")


with open('p1.txt') as f:
    lines = f.readlines()

ecgX = []
ecgY = []
for i in range(len(lines) - 1):
    L = lines[i + 1].split()
    ecgX.append(int(L[0]))
    ecgY.append(int(L[1]))

test_signal_array=np.array(ecgY)
seg1 = np.array(test_signal_array[0:1000])
Acc = sm.tsa.acf(seg1, nlags=len(seg1))
DCT = scipy.fftpack.dct(Acc[0:100], 2)
res=np.array(DCT).astype(int)


for i in range(ecg_data.shape[1]):
    signal = ecg_data.iloc[i]
    subject_data = signal
    subject_array_data = np.array(subject_data).astype(int)

    if np.array_equal(res, subject_array_data):
        print(f"The ECG data matches with the subject ")
    else:
        print(f"The ECG data does not match the subject ")

