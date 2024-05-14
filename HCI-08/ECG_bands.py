from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from pywt import wavedec

EEG_File = open(r"EEG Signal.txt", "r")
lines = EEG_File.readlines()

amp = []
for i in lines:
    amp.append(int(i))

signal = np.array(amp)
signal = (signal - min(signal)) / (max(signal) - min(signal))  # normalization
coeff = wavedec(signal, "db4", level=9)  # CA9,CD9,CD8,CD7,CD6

Delta, Theta, Alpha, beta, gamma = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]

plt.figure(figsize=(12, 6))
plt.subplot(321)
plt.plot(np.arange(len(signal)), signal)
plt.xlabel("time")
plt.ylabel("Amplitude")

plt.subplot(322)
x = np.arange(len(Delta))
x = (x - min(x)) / (max(x) - min(x))
plt.plot(x, Delta)
plt.xlabel("Delta")

plt.subplot(323)
x = np.arange(len(Theta))
x = (x - min(x)) / (max(x) - min(x))
plt.plot(x, Theta)
plt.xlabel("Theta")

plt.subplot(324)
x = np.arange(len(Alpha))
x = (x - min(x)) / (max(x) - min(x))
plt.plot(x, Alpha)
plt.xlabel("Alpha")

plt.subplot(325)
x = np.arange(len(beta))
x = (x - min(x)) / (max(x) - min(x))
plt.plot(x, beta)
plt.xlabel("beta")

plt.subplot(326)
x = np.arange(len(gamma))
x = (x - min(x)) / (max(x) - min(x))
plt.plot(x, gamma)
plt.xlabel("Gamma")

plt.show()

np.random.seed(0);
my_matrix = np.random.randn(20,5)

my_model = PCA(n_components=5)
my_model.fit(my_matrix)

print(my_model.explained_variance_)
print(my_model.explained_variance_ratio_)
print(my_model.explained_variance_ratio_.cumsum())


np.set_printoptions(precision=3)

noralized_Electrodes_copy = noralized_Electrodes.copy()
noralized_Electrodes_copy -= np.mean(noralized_Electrodes_copy,axis=0);
print("B1 is B after centering:")
print(noralized_Electrodes_copy)

pca =PCA()
x=pca.fit_transform(noralized_Electrodes_copy)
print("x")
print(x)

eigenvecmat=[]
print("Eigenvectors:")
for eigenvector in pca.components_:
    if eigenvecmat==[]:
        eigenvecmat=eigenvector
    else:
        eigenvecmat=np.vstack((eigenvecmat,eigenvector))

print("eigenvector-matrix")
print(eigenvecmat)

eigenvalmat=[]
print("eigenvalmat:")
for eigenvalue in pca.explained_variance_:
    if eigenvalmat==[]:
        eigenvecmat=eigenvalue
    else:
        eigenvalmat=np.vstack((eigenvalmat,eigenvalue))

print("eigenvalue-matrix")
print(eigenvalmat)

explained_variance_ratio_=[]
print("explained_variance_ratio_:")
for EXR in pca.explained_variance_ratio_:
    if explained_variance_ratio_==[]:
        explained_variance_ratio_=EXR
    else:
        explained_variance_ratio_=np.vstack((explained_variance_ratio_,EXR))

print("explained_variance_ratio_matrix")
print()