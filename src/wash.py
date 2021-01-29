# -*- encoding: utf-8 -*-
import numpy as np
import json
import os
import sys
from pprint import pprint
import time
import codecs
import csv
import xlrd

def windnoiseToNPY():
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

def xlsx_to_csv(xlsx_path,  csv_path):  
    workbook = xlrd.open_workbook(xlsx_path)
    table = workbook.sheet_by_index(0)
    with codecs.open(csv_path, 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)          

def strtime2timedot(strtime):   # 将时间字符串转化为时间戳
    timeArray = time.strptime(strtime, "%m-%d-%Y %H:%M:%S")
    time_dot = int(time.mktime(timeArray))
    return time_dot

def wash_csv(csv_file):
    finish = []
    for index, line in enumerate(open(csv_file, "r", encoding="utf-8")):
        if ",," in line:
            continue
        Velocity,Flow,Temperature,Humidity,Area,Direction,Time,Date = line.strip().split(",")
        if len(finish) == 0:
            line = "timedot,Temperature(℃),Velocity(m/s)"
        else:
            Temperature = Temperature.replace("℃", "")
            Velocity = Velocity.replace("m/s", "")
            timedot = strtime2timedot(Date + " " + Time)
            line = str(timedot) + "," + Temperature + "," + Velocity
        finish.append(line)
    with open (csv_file, "w", encoding="utf-8") as f:
        for line in finish:
            f.write(line + "\n")

def windspeedToJSON():
    dir_path = "../data/21.1.7/windspeed/"
    for file in os.listdir(dir_path):            # 先将xls文件装换成csv文件
        if file.split(".")[-1] == "xls":
            xlsx_to_csv(dir_path + file, dir_path + file.split(".")[0] + ".csv")
    print("convert xls to csv")
    
    for file in os.listdir(dir_path):            # 然后再对csv文件做进一步的清洗
        if file.split(".")[-1] == "csv":
            wash_csv(dir_path + file)
    print("finish wash the csv")

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
    print("convert csv to json")

if __name__ == "__main__":
    # windnoiseToNPY()        # windnoise数据转换成NPY
    windspeedToJSON()       # windspeed数据转换为json