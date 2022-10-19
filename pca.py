from utils import mnist_reader
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np

fashion_: str = 'C:/Users/peter/OneDrive - Office 365 Fontys/INF4/DASC2/CODE/DASC/fashion-mnist/data/fashion/'
img_train, label_train = mnist_reader.load_mnist(fashion_, kind='train')
img_test, label_test = mnist_reader.load_mnist(fashion_, kind='t10k')

st = time.time()
print("Starting RFC")
rfc = RandomForestClassifier()
rfc.fit(img_train, label_train)
y_pred = rfc.predict(img_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(label_test, y_pred))
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# Make an instance of the Model
pca = PCA(.95)
pca.fit(img_train)
# How many components fit 95% variance
print(pca.n_components_, 'Number of Components fitting PCA of .95')

# Apply pca
img_train = pca.transform(img_train)
img_test = pca.transform(img_test)

# New RFC, measure Performance
st = time.time()
print("Starting new RFC")
rfc2 = RandomForestClassifier()
rfc2.fit(img_train, label_train)
y_pred = rfc2.predict(img_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(label_test, y_pred))
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
