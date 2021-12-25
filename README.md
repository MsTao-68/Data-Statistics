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
