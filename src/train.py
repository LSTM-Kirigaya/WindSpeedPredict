# -*- encoding: utf-8 -*-
from forward import WindSpeedRegressor
from utils.DataLoader import *
from utils.WavPkg import *
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from parl.utils import logger

X = np.load("../data/cleanDataSet/data2117.npy")
y = np.load("../data/cleanDataSet/label2117.npy")

train_X, train_y, test_X, test_y = datasetSplit(X, y, ratio=0.8)
logger.info(f"训练集有{len(train_y)}条，测试集有{len(test_y)}条")

regression = WindSpeedRegressor(
    embedding_size=150,
    hidden_size=512,
    layer_num=1,
    rnn_dropout=0.2,
    mlp_drop=0.1
)

optimizer = torch.optim.Adam(regression.parameters(), lr=1e-3)
loss_func = nn.MSELoss()

all_losses = []

regression.train()
for epoch in range(5):
    loader = DataLoader(train_X, train_y, batch_size=256)
    for b_x, b_y in loader:
        output = regression(b_x)
        loss = loss_func(output.flatten(), b_y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_losses.append(loss.item())

        logger.info(f"loss:{all_losses[-1]}")

torch.save(regression.state_dict(), "../dist/model/test.pkl")

plt.style.use("seaborn")
plt.plot(all_losses)
plt.xlabel("number of iteration")
plt.ylabel("loss")
plt.title("loss curve of model", fontsize=20)

file_name = getCurrentTime()
plt.savefig("../figure/train/" + file_name + ".png")
plt.show()