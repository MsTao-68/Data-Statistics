# !usr\bin\python
# encoding: utf-8
# Author: Tracy Tao (Dasein)
# Date: 2021/12/25
import random

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
# 简单随机抽样，并估计总体
def pd_srs(dataset, replace, n):
    '''
    利用pd中sample函数
    params:
    dataset: input
    replace: 逐个不放回抽样 or 一次性抽取
    n: 抽样样本容量,行数
    ### DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
    frac -- 抽取行的比例
    replace -- 是否为有放回抽样 True为有放回 False为未放回抽样
    weights -- 字符索引或概率数组 代表样本权重
    random_state -- int 随机数发生器种子 random_state=1 可重现过程
    axis=0抽取行 axis=1抽取列
    '''
    if len(dataset) <= n:
        return '输入有误！'
    s = dataset.sample(n=n, replace=replace, random_state=1, axis=0)
    return s

def loadData():
    return [random.randint(10,100) for i in range(20)]

def main(dataset,k):
    '''
    dataset: 总体
    k：抽样样本容量
    '''
    data = random.sample(dataset, k)
    print('简单随机抽样数据：', data)
    sample_sum = 0 # 初始化样本和
    for i in data:
        sample_sum+=i
    sample_avg = sample_sum/k
    print("总体均值估计：",sample_avg)
    print('总体总值估计：', sample_avg * len(dataset))

# 分层随机抽样
def Stratified(dataframe, col_name,fraction):
    '''
    params:
    dataframe: input
    col_name: 要分成抽样的列名
    fraction: 抽取比例
    '''
    assert col_name is not None
    assert len(dataframe) > 1
    grouped = dataframe.groupby(by = col_name)
    # print(grouped.groups) # 同于查看分组数据

    keys = dataframe[col_name].unique().tolist() # 获取组名
    s =pd.DataFrame()
    for x in keys:
        data = df[df[col_name] == x]
        sample = data.sample(int(fraction * len(data)))
        s = s.append(sample)
    return s

# 调用train_test_split进行分层抽样
def sklearn_stratified(df, col_name):
    stratified_sample, _= train_test_split(df, test_size=0.8, stratify=df[[col_name]])
    return stratified_sample

# 整群抽样
def ClusterSampling():
    pass

# 系统抽样
def Systematic(dataset, step):
    '''
    params:
    dataset: input
    step
    '''
    s = pd.DataFrame()
    for i in range(0, len(dataset), step):
        s = pd.concat([s,dataset.iloc[i,:]])
    return s

if __name__ == '__main__':
    dataset = loadData()
    main(dataset,10)
    df = pd.read_csv('E:\\2021-2022第一学期\\Tableau\\20211115\\订单与商品APAC.csv',header=0)
    # print(Stratified(df,'邮寄方式', 0.2).head())
    # print(sklearn_stratified(df,'邮寄方式'))
    print(Systematic(df,100).T.head())

