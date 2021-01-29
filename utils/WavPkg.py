import time
import sys
import numpy as np
import json
import os
import datetime
from pprint import pprint
import matplotlib.pyplot as plt

sample_rate = 22050    # 项目的采样率

def timedot2strtime(timedot):   # 将时间戳转化为系统中的时间字符串，比如17.10.2020 09:45:58
    if isinstance(timedot, str):
        timedot = int(timedot)
    local_time = time.localtime(timedot)
    std_time = time.strftime("%d.%m.%Y %H:%M:%S", local_time)
    return std_time

def strtime2timedot(strtime):   # 将时间字符串转化为时间戳
    timeArray = time.strptime(strtime, "%d.%m.%Y %H:%M:%S")
    time_dot = int(time.mktime(timeArray))
    return time_dot

def timedotIndexMap(begin, td):       # 将时间戳映射到索引值(0-(1323000-1))
    if td - begin > 60:
        raise ValueError("超出映射范围")
    return (td - begin) * sample_rate

def getCurrentTime(str_format="%Y-%m-%d-%H-%M-%S"):       # 获取当前的时间，一般是为了命名文件
    return datetime.datetime.now().strftime(str_format)

def ProcessOneMinuteNPY(npy_file : str, speed_dict : str) -> [np.ndarray, np.ndarray]:    # 针对风速表将对应的数据和真值合并输出
    data = np.load(npy_file, allow_pickle=True)
    sample_num = data.shape[-1]         # 读取的波形文件的采样点数量
    if sample_num % sample_rate != 0:
        raise ValueError(f"确保输入的波形文件的采样点数量为采样率{sample_rate}的倍数，但是接收到的波形文件的采样点数量为{sample_num}")

    # 查询各个子区间的统计的风速
    # 首先先确定需要查询的30个时间戳
    file_name = os.path.split(npy_file)[-1]     # 需要获取的30个点的第一个时间戳就是文件名，此处只是稍做处理
    begin = int(file_name.split(".")[0])
    end = begin + 60
    wav_arrays, speeds = [], []
    for td in range(begin + 1, end):   # 考虑到风速仪测风速的滞后性，此处将第t秒的回归值对应到第t-1秒到第t秒的风速
        if str(td) in speed_dict:    # 如果在字典中，则记录
            oneSecondWav = data[..., timedotIndexMap(begin, td - 1) : timedotIndexMap(begin, td)]
            wav_arrays.append(oneSecondWav)
            speeds.append(speed_dict[str(td)]["speed"])

    return wav_arrays, speeds

if __name__ == "__main__":
    # speed_dict_path = "../data/dict/speed_dict.json"
    # with open(speed_dict_path) as f:
    #     speed_dict = json.load(f)
    #
    # wavs, speeds = ProcessOneMinuteNPY(
    #     npy_file="../data/.npy/1602899326.npy",
    #     speed_dict=speed_dict
    # )
    #
    # print(len(wavs))
    # print(len(speeds))
    print(getCurrentTime())
    with open("../data/dict/21.1.7/speed_dict.json") as f:
        speed_dict = json.load(f)

    print("样本中的1秒数据段个数：", len(os.listdir("../data/.npy")) * 60)
    print("字典中的label个数：", len(speed_dict))

    values = []
    for value in speed_dict.values():
        values.append(int(value["speed"]))

    plt.style.use("seaborn")
    plt.plot(values)
    plt.show()

