# WindSpeedPredict

wind speed prediction based on DNN.

---

## 预备工作

`requirements.txt`中为需要安装的库，运行`pip install -r .\requirements.txt`即可安装完成。需要说明的是，本项目的深度学习框架为PyTorch，请前往官网下载符合版本需求的cpu版pytorch

如果你是第一次克隆，在根目录创建data文件夹，其中内容如下：

```bash
data
  -|.npy
  -|21.1.7
    -|windnoise
        -|...
    -|windspeed
        -|...
   -|cleanDataSet
   -|dict
```

`21.1.7`是小组提供的风速数据集，请确保windspeed中的数据是xls文件。

在拖入数据集后，在主目录中进入命令行，输入如下指令：

```bash
cd .\src\
python -u .\wash.py
```

运行后，你会在`..\data\.npy`中得到汇整成NPY格式的波形采样文件，在`..\data\dict`中得到整理好的json文件。
