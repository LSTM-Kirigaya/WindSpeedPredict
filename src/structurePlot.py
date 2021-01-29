# -*- encoding: utf-8 -*-
from forward import WindSpeedRegressor
import torch
import hiddenlayer as h

model = WindSpeedRegressor(
    embedding_size=150,
    hidden_size=512,
    layer_num=1,
    rnn_dropout=0.2,
    mlp_drop=0.1
)

vis_graph = h.build_graph(model, torch.zeros([1, 2, 22050]))  # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
vis_graph.save("../figure/structure/WindSpeedRegressor.png")  # 保存图像的路径