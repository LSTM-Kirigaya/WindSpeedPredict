# -*- encoding: utf-8 -*-
from utils.WavPkg import *
import numpy as np
import json
import os
from pprint import pprint
import sys

with open("../data/dict/21.1.7/speed_dict.json") as f:
    speed_dict = json.load(f)

npy_files = os.listdir("../data/.npy/")
npy_paths = list(map(lambda x : "../data/.npy/" + x, npy_files))

duplicate = 0
all_wav_arrays, all_speeds = [], []
for path in npy_paths:
    try:
        wav_arrays, speeds = ProcessOneMinuteNPY(path, speed_dict)
        all_wav_arrays += wav_arrays
        all_speeds += speeds
    except:
        duplicate += 1

all_wav_arrays = np.array(all_wav_arrays)
all_speeds = np.array(all_speeds)

print("数据集重构率：", (len(npy_files) - duplicate) / len(npy_files))
print("波形数据shape：", all_wav_arrays.shape)
print("标签shape：", all_speeds.shape)

for ndarray, file_name in zip([all_wav_arrays, all_speeds], ["data2117", "label2117"]):
    np.save("../data/cleanDataSet/" + file_name, ndarray)