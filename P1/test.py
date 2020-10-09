#!/usr/bin/env python3

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from MySVC import MySVC
import matplotlib.pyplot as plt
from Utils import *
import time

seed = 123

x, y = make_moons(noise=1e-1, random_state=seed)
y[y != 1] = -1

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed)

plot_dataset_clas(x_tr, y_tr)
plt.title('Train')
plt.show()
plot_dataset_clas(x_te, y_te)
plt.title('Test')
plt.show()

C = 1
gamma = 'scale'

model_my = MySVC(C=C, gamma=gamma, seed=seed)
model_sk = SVC(C=C, gamma=gamma)

# Training of the models (complete).
t = time.process_time()
model_my.fit(x_tr, y_tr)
print("Training time en MySVC: ", time.process_time() - t)
t = time.process_time()
model_sk.fit(x_tr, y_tr)
print("Training time en Sklearn: ", time.process_time() - t)


predict_my = model_my.predict(x_te)
predict_sk = model_sk.predict(x_te)

# plot_svc(x_tr, y_tr, model_sk)
# Comparative of the predicted scores (complete).

score_my = model_my.score(x_te, y_te)
score_sk = model_sk.score(x_te, y_te)
print("Score en MySVC: ", score_my)
print("Score en Sklearn: ", score_sk)

# Comparative of the predicted classes (complete).
# ...
