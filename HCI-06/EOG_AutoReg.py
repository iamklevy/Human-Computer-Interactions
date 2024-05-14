import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from statsmodels.tsa.ar_model import AutoReg

H_file_wb = openpyxl.load_workbook("Horizontal Signals.xlsx")
H_file_ws = H_file_wb['Sheet1']

cols = list(H_file_ws.columns)
num_cols = int(H_file_ws.max_column)

for i in range(num_cols):
    col_range = cols[i]
    signal = []
    for a in col_range:
        signal.append(a.value)
    print(signal)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(0, len(signal)), signal)
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.show()
    model = AutoReg(signal, lags=4)
    model_fit = model.fit()
    print('Coefficients: %s' % model_fit.params)




