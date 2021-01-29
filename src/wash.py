# -*- encoding: utf-8 -*-
import numpy as np
import json
import os
import sys
from pprint import pprint

# 采样点转化成.npy文件
wind_file = []
for file_name in os.listdir("../data/21.1.7/windnoise"):
    if file_name == "解释.txt":
        continue
    if file_name.split('.')[-1] == "txt":
        wind_file.append(file_name)

for file in wind_file:
    path = "../data/21.1.7/windnoise/" + file
    for idx, line in enumerate(open(path)):
        name = line.split(":")[0]
        data = line.split()[1:]
        data_ndarray = np.array(data, dtype="int64")
        left_channel = data_ndarray[:-1:2]
        right_channel = data_ndarray[1::2]
        print(f"{name}.npy done")
        np.save(f"../data/.npy/{name}.npy", np.array([left_channel, right_channel]))

"""
.npy文件说明：
index in [0, 19], 无背景噪音，无人声的风噪数据
index in [20], 塔扇本身的噪音
index in [20, 40], 有背景噪音，无人声的风噪数据
index in [40, 60], 背景噪声文件
"""

# 将风速文件整合转化为.json文件
wind_file = []
for file_name in os.listdir("../data/21.1.7/windspeed"):
    if file_name.split('.')[-1] == "csv":
        wind_file.append(file_name)

speed_dict = {}
for file in wind_file:
    path = "../data/21.1.7/windspeed/" + file
    for idx, line in enumerate(open(path, "r", encoding="utf-8")):
        if idx == 0:
            continue
        else:
            spt = line.split(",")
            if len(spt) == 3:
                time, temperature, speed = line.split(",")
                speed_dict[time] = {
                    "temperature" : float(temperature),
                    "speed" : float(speed)
                }
            else:
                continue

with open("../data/dict/21.1.7/speed_dict.json", "w", encoding="utf-8") as f:
    json.dump(speed_dict, f, ensure_ascii=False)
    print(f"speed_dict.json done")