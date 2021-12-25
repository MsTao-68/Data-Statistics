# !usr\bin\python
# encoding: utf-8
# Author: Tracy Tao (Dasein)
# Date: 2021/12/25

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
# %matplotlib inline

from scipy.interpolate import interp1d
# 准备数据
import pandas as pd
gdp = pd.read_csv('E:\\dasein_py\\Tesla\\raw data\\GDP2.csv',header = 0, index_col=0)
gdp = gdp.reset_index()
q = 8
m = (q - 1) * 3 + 1
x = np.linspace(1,m,q)
# print(x)
xx = np.linspace(1,m,m)
# print(xx)
y = gdp['国内生产总值 绝对值(亿元)']

# 绘制数据
plt.figure(figsize=(15,15),dpi=80)
plt.subplot(3,3,1)
# plt.scatter(x,y,s=50, color ="#00338D")
plt.plot(x,y,color ="#00338D",marker = '*',
         label ='example')
plt.title("Example")
plt.legend()

plt.subplot(3,3,2)
# 线性插值方式
linear = si.interp1d(x,y,kind = 'linear')
y1 = si.interp1d(x,y,'linear')
linear_x = np.linspace(1, m)
linear_y = y1(linear_x)
plt.plot(linear_x, linear_y,
         color="#00338D",marker='*',
         label="linear interp1d")
plt.title('Linear interp1d')
plt.legend()

# 三次样条插值
plt.subplot(3,3,3)
cubic = si.interp1d(x,y,kind='cubic')
zx = np.linspace(1, m)
cubic_y = cubic(zx)
plt.plot(zx, cubic_y,
         color="#00338D",marker='*',
        label="cubic interp1d")
plt.title('Cubic')
plt.legend()

#邻近插值
plt.subplot(3,3,4)
y2 = si.interp1d(x,y,'nearest')
nearest_x = np.linspace(1, m)
nearest_y = y2(nearest_x)
plt.plot(nearest_x,nearest_y,
         color="#00338D", marker='*',
         label="nearest interp1d")
plt.title('Nearest')
plt.legend()

# slinear
plt.subplot(3,3,5)
y3 = si.interp1d(x,y,'slinear')
slinear_x = np.linspace(1, m)
slinear_y = y3(slinear_x)
plt.plot(slinear_x,slinear_y,
         color="#00338D",marker='*',
         label="slinear interp1d")
plt.title('slinear')
plt.legend()

# previous
plt.subplot(3,3,6)
y4 = si.interp1d(x,y,'previous')
previous_x = np.linspace(1, m)
previous_y = y4(previous_x)
plt.plot(previous_x,previous_y,
         color="#00338D", marker='*',
         label="previous interp1d")
plt.title('previous')
plt.legend()

# quadratic
plt.subplot(3,3,7)
y6 = si.interp1d(x,y,'quadratic')
qx = np.linspace(1, m)
qy = y6(qx)
plt.plot(qx,qy,
         color="#00338D", marker='*',
         label="quadratic interp1d")
plt.title('quadratic')
plt.legend()
plt.show()
