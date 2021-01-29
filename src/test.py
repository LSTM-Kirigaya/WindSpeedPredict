import torch
import numpy as np
from utils.DataLoader import *
from utils.WavPkg import *
from forward import WindSpeedRegressor
import matplotlib.pyplot as plt
import datetime

X = np.load("../data/cleanDataSet/data1017.npy")
y = np.load("../data/cleanDataSet/label1017.npy")

train_X, train_y, test_X, test_y = datasetSplit(X, y, ratio=0.8)

model = WindSpeedRegressor(
    embedding_size=150,
    hidden_size=512,
    layer_num=1,
    rnn_dropout=0.2,
    mlp_drop=0.1
)

model.load_state_dict(torch.load("../dist/model/test.pkl"))

model.eval()
out = model(train_X).flatten()
mean_error = torch.mean(out - train_y)
print("平均误差为：{:.3f} m/s".format(mean_error.item()))

plt.style.use("seaborn")
plt.plot(out.data.numpy()[:10], label="predict")
plt.plot(train_y.data.numpy()[:10], label="true ground")
plt.legend()

file_name = getCurrentTime()
plt.savefig("../figure/test/" + file_name + ".png")
plt.show()