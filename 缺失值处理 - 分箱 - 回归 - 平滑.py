# !usr\bin\python
# encoding: utf-8
# Author: Tracy Tao (Dasein)
# Date: 2021/12/25

# KF 分箱
'''
算法流程：
1. 初始化：根据连续变量的值排序，离散化
2. 合并
3. repeat
4. 计算每个相邻的箱子的卡方值
5. 对低卡方值（类相似）的箱子合并
6. until
7. 卡方值 >= threshold，显著性水平=。1/.05/.01 or 最大最小分箱数量
'''
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
# 调包平滑tsmoothie
from tsmoothie.utils_func import  sim_randomwalk
from tsmoothie.smoother import LowessSmoother # 局部加权平滑

def bins(lower, width, quantity):
    '''
    创建等距分箱: 将连续变量分段，分成等距离的左闭右开小区间
    '''
    bins =[]
    for low in range(lower, lower + quantity * width + 1, width):
        bins.append((low, low + width))
    return bins

def find_bins(test, bins):
    '''
    查找分箱：将对应的数据点分入箱内
    '''
    for i in range(0, len(bins)):
        if bins[i][0] < test <=  bins[i][1]:
            return
        else: return -1

def counter(arr):
    '''
    统计值的频率
    '''
    f = Counter(arr)
    return f

# 回归填充法

def regression_fillna(dataset, col_name, K):
    '''
    params:
    dataset: original dataframe
    col_name: the columns that needs to be fillna
    K: 相关系数最大前k个
    return: concat dataframe
    '''
    df_test = dataset[np.isnan(dataset[col_name])] # 只含缺失值子集
    print('The quantity of column {} is {}.'.format(col_name, df_test.shape[1]))
    df_train = dataset.dropna(col_name, axis=0) # 不含缺失值子集
    p_test = np.array(df_train[col_name].T)
    score, p_value = ss.normaltest(p_test)
    if p_value <= 0.05:
        print('p_value = {}, 缺失值数据非正态分布，不能回归预测填充。'.format(p_value))
        return
    print('p_value = {}, 缺失值数据正态分布，能回归预测填充。'.format(p_value))
    print('故进行变量相关性检验。')
    cor = dataset.corr()
    cols = (cor.nlargest(K, col_name)[col_name].index).values
    lr = LinearRegression() # 建立线性回归模型进行预测，并填充缺失值
    lr.fit(df_train[cols], df_train[col_name])
    y_pred = lr.predict(df_test)
    df_test[col_name] = y_pred
    # 填充完进行合并数据集
    df_all = df_train.concat(df_test, ignore_index= True)
    return df_all

def Lowess_smoother(n,length):
    '''
    利用参数生成随机数组
    '''
    data = sim_randomwalk(n_series=n,timesteps=length,process_noise=10, measure_noise=30)
    smoother = LowessSmoother(smooth_fraction=0.1, iterations=1) # 平滑初始化
    smoothed = smoother.smooth(data)
    return  smoothed # 返回平滑后的数据







if __name__ == "__main__":
    # 实例：学生成绩分箱
    score = np.random.randint(30,100,size = 20)
    bins = [0, 60, 70, 80, 90, 100]
    cat = pd.cut(score, bins)
    print(Counter(cat))
    print(pd.value_counts(cat))

