from utils import mnist_reader
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.decomposition import PCA
import numpy as np

fashion_: str = 'C:/Users/peter/OneDrive - Office 365 Fontys/INF4/DASC2/CODE/DASC/fashion-mnist/data/fashion/'
X_train, y_train = mnist_reader.load_mnist(fashion_, kind='train')
X_test, y_test = mnist_reader.load_mnist(fashion_, kind='t10k')
st = time.time()
print("Starting RFC")
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

pca = PCA(n_components=2)
