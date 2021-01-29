import torch
import json
import numpy as np

def datasetSplit(X, y, ratio=0.8, X_type=torch.float32, y_type=torch.float32, random_seed=100):      # 分割原数据得到训练集和测试集
    """
    :param X: 数据
    :param y: 标签
    :param ratio: 划分为训练集的比例
    :return:
    """
    sample_num = len(y)
    # 先打乱
    np.random.seed(random_seed)
    index = np.arange(sample_num)
    np.random.shuffle(index)

    offline = int(sample_num * ratio)

    train_X = torch.tensor(X[index[:offline]], dtype=X_type)
    train_y = torch.tensor(y[index[:offline]], dtype=y_type)
    test_X = torch.tensor(X[index[offline:]], dtype=X_type)
    test_y = torch.tensor(y[index[offline:]], dtype=y_type)

    return train_X, train_y, test_X, test_y

def DataLoader(X, y, batch_size, X_type=torch.float32, y_type=torch.float32, shuffle=True):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=X_type)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=y_type)

    sample_num = len(y)  # 样本数量
    index = np.arange(sample_num)
    if shuffle:
        np.random.shuffle(index)
    b_x, b_y = [], []   # 存放一个batch的数据和标签
    for i in index:
        b_x.append(X[index[i]].data.numpy().tolist())
        b_y.append(y[index[i]].data.numpy().tolist())
        if len(b_x) == batch_size:
            yield [
                torch.tensor(b_x, dtype=X_type),
                torch.tensor(b_y, dtype=y_type)
            ]
            b_x.clear()
            b_y.clear()
    # yield出剩余的数据
    yield [
            torch.tensor(b_x, dtype=X_type),
            torch.tensor(b_y, dtype=y_type)
        ]