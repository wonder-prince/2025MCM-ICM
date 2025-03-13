import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


def monte_carlo_shapley_values(X, y, model=LinearRegression(), m=1000):
    """
    计算每个特征的Shapley值

    X: 特征数据，大小为 (N, M)
    y: 目标数据，大小为 (N,)
    model: 用于拟合的回归模型（可替换为其他模型）
    m: 蒙特卡洛采样次数
    """
    N, M = X.shape  # N是样本数，M是特征数
    shapley_values = np.zeros(M)  # 初始化每个特征的Shapley值

    # 创建一个包含所有特征索引的列表
    feature_indices = list(range(M))

    # 定义价值函数 v(S)，根据特征子集 S 来训练模型并评估性能
    def value_function(S):
        # S 是特征索引的集合
        X_subset = X[:, S]  # 提取当前特征子集
        model.fit(X_subset, y)  # 使用特征子集训练模型
        y_pred = model.predict(X_subset)  # 预测目标值
        return mean_squared_error(y, y_pred)  # 使用均方误差作为性能度量

    # 蒙特卡洛采样过程
    for _ in range(m):
        # 随机排列特征索引
        random_order = random.sample(feature_indices, M)
        current_value = 0  # 当前模型性能（误差）

        # 对于每个特征，在当前顺序中计算其边际贡献
        for i in range(M):
            feature_i = random_order[i]
            # 计算当前特征的边际贡献：加入该特征后的模型性能变化
            current_contribution = value_function(random_order[:i + 1]) - current_value
            shapley_values[feature_i] += current_contribution
            current_value = value_function(random_order[:i + 1])

    # 归一化Shapley值
    shapley_values /= m

    return shapley_values


df = pd.read_excel('with_medal.xlsx')
total_num = pd.read_excel('total_counts.xlsx')
# 输入参数
nation_name = "China"
nation_mini = "CHN"

total_nation = total_num[total_num['NOC'] == nation_name].drop(columns=[total_num.columns[0]])
total_nation = total_nation.values
df_nation = df[df['NOC'] == nation_mini]
# 按年份和项目进行分组，并统计每种奖牌的数量
result = df_nation.groupby(['Year', 'Sport']).size().unstack(fill_value=0)
# 显示结果
print(result)
print(total_nation)
shapley_values = monte_carlo_shapley_values(result, total_nation.flatten())
print("Shapley values for each feature:", shapley_values)