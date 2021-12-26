# Data-Statistics
Codes for data preprocessing before creating models by using Statistics knowleges.
---
- <font color='#00338D'>灵感来源于：大数据探索性分析，认识到数据探索性分析以及数据预处理对构建模型的重要性。</font></div>
-  <div align="center"><font color='red'>统计学很重要！！！</font></div>
---
## 相关概念引入，用于解释Data ∞ Statistics 的逻辑和supporting。
1. Data Mining 概念
- Mine out 隐含、以前未知、潜在有用的信息
- Use 自动或半自动的方法、数据探索发现有用pattern （**Quote from：Bishop - Pattern Recogniction**）
- Knowledge discovery in Database
- 利用各种技术与统计方法，将大量的历史数据**整理分析**、归纳、整合，从海量数据中挖掘有用的隐藏信息：*趋势、特征、相关性*
2. 数据挖掘步骤中数据探索性分析是步骤之一
- Define Question: classification/regression/clustering
- Create db: data collection, description, feature selection, evaluate, data cleaning, merge, 构建元数据， 加载数据挖掘库， 维护数据库
- **Data analysis** 
- **Data preprocessing**: feature selection, 选择记录， 创建新变量， 转换新变量
- 评价变量，解释模型
---
## 数据预处理
### 数据清洗
#### 异常值处理方式
1. 分箱
   - 目的：减少数据量；连续变量离散化采用分箱法
   - 方法
     - 有监督分箱
       1. [x] 卡方分箱（自底向上）：基于卡方检验的值，低卡方值=类分布相似
          - 公式：X ^ 2  =  sumi ( sumj ( ( Aij - Eij ^ 2 ) / Eij ) )
       2. [ ] 最小熵分箱
          - 公式： sumk (sumj ( -pij * logpij ) )
     - 无监督分箱
       1. [x] 等距分箱
       2. [ ] 等频率分箱
2.  [x] 回归
   - 一般把异常值当成缺失值处理：删除、统计填充值（均值、中位数等）、回归方程预测填充
     - 缺点：虽然无偏，但是会忽视随机误差，低估标准差和其他未知量的测量值。
     - 注：使用前，必须假设存在缺失值所在的变量与其他变量是线性相关，但实际不一定存在线性关系
   - 步骤
     1. 确定填充缺失值的变量（特征列）
     2. 拆分：原始数据集 = 不含缺失值 + 只含缺失值
     3.  研究变量相关性
     4. 不含缺失值的数据集建立回归模型，预测只含缺失值的数据集
     5. 合并数据集
3. [x] 平滑 tsmoothie
   - 针对时间序列数据
     1. 指数平滑
     2. 具有窗口类型的卷积平滑（常数、汉宁、汉明、巴特利特、布莱克曼）
     3. 傅里叶变换的频谱平滑
     4. 多项式平滑
     5. 样条平滑（线性、三次、自然三次）
     6. 高斯平滑
     7. 二进制平滑
#### 缺失值处理方式之插值法
1. 插值函数：
   - 补全缺失数据
   - 若是时间序列数据，用于生成其他时间点数据，以增大数据量
   - 图像处理处理中，用插值的方式补全像素点，提高图像精度或者清晰率
     - 插值方式：线性插值；三系样条插值
   - [x] Scipy.interpolate
      - 用于处理GDP数据处理：国家统计局上官方只有季度数据，希望用插值法将季度GDP数据转为月度GDP数据
         - 注：效果不是很好，并且部分参数使用后有越过年终GDP数据的现象，**待更新：季度GDP数据转为月度GDP数据**。

---
### 抽样
2021/12/25 upload 🦾
- 有效抽样的主要原理是：如果抽样是有代表性的，则使用抽样样本估计总体。注：代表性：抽样样本近似地具有与元数据集相同的（感兴趣的）性质。
- 概率抽样方法：（房祥忠《大数据探索性分析》）
  1. [x] 简单随机抽样 Simple Random Sampling
  2. [x] 分层随机抽样 Stratified Random Sampling
  3. [ ]整群抽样、聚类抽样 Cluster Sampling 🧠 **Work on Progress**
  4. [x] 系统抽样 Systematic Sampling
  5. [ ]多阶段采样 🦾🧠 **Work on Progress**
- 非概率抽样方法：
  1. 便利抽样
  2. 滚雪球抽样
  3. 判断抽样
  4. 配额抽样
---

## 数据分析方法
1. [x] 因子分析法（Python版） 
##### 主成分分析和因子分析对比
- 原理对比
    - PCA: 寻找原有自变量的线性组合，取出线性关系影响较大的原始数据作为主成分
    - FA: 使得所有自变量可以通过若干个因子（中间变量）被观察到
- 数据分析流程
   1. 数据标准化
   2. Adequancy Test（充分性检测：KMO & 巴特利特球形检测）
   3. 协方差矩阵（相关系数矩阵）
   4. 特征值 & 特征向量
   5. 主成分个数（因子）
   6. 方差，贡献率，累计贡献率
   7. 因子分析
- packages(调包很简单，原理🤯）
  - import numpy as np
  - import pandas as pd
  - from sklearn.preprocessing import MinMaxScaler, StandardScaler
  - import numpy.linalg as nlg
  - from factor_analyzer import factor_analyzer,Rotator
  - from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity,factor_analyzer,Rotator
2. [x] 主成分分析法（Python版）
##### Principal Component Analysis 
- 综述：数据降维并尽可能减少信息损失，主成分就是方差贡献率大的自变量
- 理论：通过正交变换将一组可能存在相关性的自变量转换为线性不相关的自变量（即主成分）
- 步骤：
    0. repeat{
    1. Data overview & preprocessing, 去除可以分析出来的不相关列以及影响不大的列
    2. 计算相关系数矩阵或者协方差矩阵，取出n_largest的自变量作为主成分
        - 注：数据标准化后用corr(), 数据未标准化用cov()
    3. 判断是否存在明显的多重共线性
        - 注：多重共线性：自变量间存在一个变量可以表示为多个变量的线性组合
        - VIF方差膨胀系数：statsmodels.stats.outliers_influence.variance_inflation_factor() }until VIF < 0.7
    4. 主成分适应性检测：KMO , batelette球形检测
    5. 得到主成分表达式并确定主成分个数，选取主成分
- Packages:
   - import pandas as pd
   - import numpy as np
   - from numpy.linalg import eig
   - from sklearn.datasets import load_iris
   - import matplotlib.pyplot as plt
   - from factor_analyzer import factor_analyzer,Rotator
   - from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
3. [x] 独立成分分析法 （Python版）Independent Components Analysis
   1. Background: 
       1. 鸡尾酒会混合音源分离提取；数据降维
       2. 假设：
          - 成分独立
          - 线性混合
          - 非高斯性分布
              1. Kurtosis 四阶累积量
              2. Negentropy 负熵
              3. 互信息最小化
       3. 不确定性
   3. 对比PCA
       - PCA: 降维，但对高斯分布样本有效；ICA: 降维，样本服从非高斯性分布
       - PCA将数据映射到新的低维空间，并且各维度不相关；ICA是寻找数据可能成分组，用于提取特征
   4. 数据分析流程
       1. 中心化和白化（减少特征相关性；特征具有相同方差（即协方差阵是单位阵))
       2.  - **Source from：WeChat Public Account: 脑机接口研习社**：脑电图和脑磁图(EEG/MEG)的数据分析
   ![output_12_1](https://user-images.githubusercontent.com/84648756/147399697-54aba29d-dd82-4d0b-9676-123b9ccb7c07.png)
   ![output_9_0](https://user-images.githubusercontent.com/84648756/147399707-0c5b35fc-096a-4dd9-b978-e0edf917a7d6.png)



   
