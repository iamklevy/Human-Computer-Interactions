import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

file = pd.read_csv(r"multi_chanel.csv")
electrodes = []
for i in range(file.shape[1]):
    data = []
    for j in range(file.shape[0]):
        data.append(file.iloc[j, i])

    electrodes.append(data)

normalized_Electrodes = preprocessing.normalize(electrodes, axis=0)

plt.figure(figsize=(12, 6))
plt.subplot(411)
plt.plot(electrodes[0])
plt.subplot(412)
plt.plot(electrodes[1])
plt.subplot(413)
plt.plot(electrodes[2])
plt.subplot(414)
plt.plot(electrodes[3])
plt.show()




