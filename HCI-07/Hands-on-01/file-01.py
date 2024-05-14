import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel("Horizontal Signals.xlsx", header=None)


def calculate_auc(signal):
    return np.trapz(signal)


means = []
stds = []
aucs = []

for i in range(data.shape[1]):
    signal = data[i]
    mean = np.mean(signal)
    std = np.std(signal)

    auc = calculate_auc(signal)

    means.append(mean)
    stds.append(std)
    aucs.append(auc)

features_df = pd.DataFrame({'mean': means, 'std': stds, 'AUCs': aucs})

with pd.ExcelWriter("Features.xlsx") as writer:
    features_df.to_excel(writer, sheet_name="sheet01", index=False)
