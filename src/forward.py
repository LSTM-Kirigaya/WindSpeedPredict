# -*- encoding: utf-8 -*-
import torch
from torch import nn
import numpy as np

class WindSpeedRegressor(nn.Module):
    def __init__(self, embedding_size, hidden_size, layer_num, rnn_dropout=0, mlp_drop=0, sample_num=22050):
        """
        :param embedding_size: 嵌入维度，也就是我们加窗片段的采样点数量
        :param hidden_size: RNN算子的隐层
        :param layer_num: RNN的层数
        :param dropout: 每个cell被丢弃的概率
        :param sample_num: 输入待回归波形的采样点数量
        """
        self.embedding_size = embedding_size    # 加窗的宽度
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.dropout = rnn_dropout
        self.sample_num = sample_num

        super(WindSpeedRegressor, self).__init__()
        # 前馈的几个算子
        self.LGru = nn.GRU(             # 学习左声道的RNN
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            dropout=0 if layer_num == 1 else rnn_dropout,
            batch_first=True,
            bidirectional=False
        )

        self.RGru = nn.GRU(             # 学习右声道的RNN
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            dropout=0 if layer_num == 1 else rnn_dropout,
            batch_first=True,
            bidirectional=False
        )

        self.regression = nn.Sequential(
            nn.Linear(2 * hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Dropout(mlp_drop),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(mlp_drop),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def wavEmbedding(self, wav):       # 将波形嵌入矩阵
        """
        :param wav: shape[batch, 2, sample_num]
        :return: [batch, time_step, embedding_size]与[batch, time_step, embedding_size]
        """
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav, dtype=torch.float32)
        if len(wav.shape) != 3:
            raise ValueError(f"请确保输入前馈网络的数据是三阶张量，但是本次输入的数据的shape为{wav.shape}")
        batch = wav.shape[0]
        sample_num = wav.shape[-1]
        emb_size = self.embedding_size
        time_step = sample_num // emb_size
        # TODO 解决sample_num无法被emb_size整除的情况

        # 切分并重构左声道与右声道的波形向量
        left = wav[:, 0, :]
        right = wav[:, 1, :]

        return [
            left.reshape([batch, time_step, emb_size]),
            right.reshape([batch, time_step, emb_size])
        ]


    def forward(self, wav):
        left, right = self.wavEmbedding(wav)

        # TODO 尝试取RNN最后一位，RNN所有output取平均等等方法来实验

        left_output, _ = self.LGru(left)
        right_output, _ = self.RGru(right)

        cat_rnn_output = torch.cat([left_output[:, -1, :], right_output[:, -1, :]], dim=1)      # 将左声道和右声道的output的最后一位拿出来做拼接

        output = self.regression(cat_rnn_output)
        return output



if __name__ == "__main__":
    data = np.load("../data/cleanDataSet/data1017.npy", allow_pickle=True)
    labels = np.load("../data/cleanDataSet/label1017.npy", allow_pickle=True)

    regressor = WindSpeedRegressor(
        embedding_size=150,
        hidden_size=512,
        layer_num=1,
        rnn_dropout=0.2
    )

    print(regressor(data[:10]))
    print(labels[:10])