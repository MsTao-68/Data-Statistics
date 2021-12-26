# !usr\bin\python
# encoding: utf-8
# Author: Tracy Tao (Dasein)
# Date: 2021/12/26

'''
因子分析算法流程
1. 数据标准化
2. Adequancy Test
3. 协方差矩阵（相关系数矩阵）
4. 特征值&特征向量
5. 主成分个数（因子）
6. 方差，贡献率，累计贡献率
7. 因子分析
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy.linalg as nlg
from sklearn.decomposition import PCA
from factor_analyzer import factor_analyzer,Rotator
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

df = pd.read_excel('E:\\dasein_py\\Data Mining and Machine Learning\\Data Analysis\\上市公司数据 (1).xlsx')
# print(df)
# scaler = StandardScaler()
# data = pd.DataFrame(scaler.fit_transform(df.iloc[:,2:]))
'''
不用Scaler,因为还需要手动转化成df，所以直接手动矩阵加减法
'''
df.fillna(0,inplace=True)
df= df.iloc[:,2:]
df = (df-np.mean(df))/np.var(df) # 手动标准化，期望为1，方差为0
print(df.head())
'''
# 充分性检测
# 巴特利特球形加测：检测变量之间是否具有关联，若不显著，就不能因子分析
# KMO 检测
'''
kmo_per_variable,kmo_total  = calculate_kmo(df)
print('因子kmo：',kmo_per_variable)
print('总体kmo: ',kmo_total) # 需要小于0.7
chi_square_value,p_value=calculate_bartlett_sphericity(df)
print('巴特利特球形检测p值：', p_value) # 要小于0.5
# 计算协方差矩阵
cov = df.corr() # 注：标准化之后相关系数矩阵就是原始数据的协方差矩阵
print('协方差矩阵：',cov)
'''
计算特征值和特征向量，并特征值可视化
'''

e = nlg.eig(cov)[0]
print('特征向量：',e)
# 特征值 eigenvalues，手动设定需要3个主成分
fa = FactorAnalyzer(3, rotation='Varimax', method='principal')
fa.fit(df)
original_eigen_values ,common_factor_eigen_values= fa.get_eigenvalues()
print('原始特征值：',original_eigen_values)
print('共同因子特征值：',common_factor_eigen_values)
import matplotlib.pyplot as plt
plt.scatter(range(1,df.shape[1]+1),original_eigen_values)
plt.plot(range(1,df.shape[1]+1),original_eigen_values)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
print("公因子方差: " ,fa.get_communalities())
print('-----------------------------------')
print('成分矩阵：',fa.loadings_)
print('-----------------------------------')
print('累计贡献率：',fa.get_factor_variance())