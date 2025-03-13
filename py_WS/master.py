import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
USA_G1= {
    'Year': [1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016],
    'Observation': [0, 1, 0, 0, 0, 0, 0, 0, 0, 15, 1, 7, 9, 1, 12, 17, 12, 21]
}

ROU_G1 = {
    'Year': [1952, 1956, 1960, 1964, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016],
    'Observation': [0, 2, 1, 0, 0, 15, 14, 16, 17, 10, 11, 11, 15, 4, 6, 0]
}
CHN_V1 = {
    'Year': [1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    'Observation': [3, 1, 0, 2, 0, 3, 1, 0, 3, 0, 0]
}
USA_V1 = {
    'Year': [1964, 1968, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    'Observation': [0, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 3, 2]
}
USA_G = pd.DataFrame(USA_G1)
ROU_G = pd.DataFrame(ROU_G1)
CHN_V = pd.DataFrame(CHN_V1)
USA_V = pd.DataFrame(USA_V1)
intervention_year1 = 1984
intervention_year2 = 1984
intervention_year3 = 2012
intervention_year4 = 2012

# 为每个数据集进行分段回归分析的函数
def segmented_regression(df, intervention_year):
    # 创建干预变量 (intervention) 和时间变量 (post)
    df['intervention'] = (df['Year'] >= intervention_year).astype(int)
    df['time'] = df['Year'] - df['Year'].min()
    df['post'] = df['Year'] - intervention_year
    df['post'] = df['post'].apply(lambda x: x if x >= 0 else 0)

    # 回归模型设计矩阵
    X = np.column_stack([np.ones(df.shape[0]), df['time'], df['intervention'], df['time'] * df['post']])
    X = sm.add_constant(X)  # 添加常数项

    # 回归分析
    Y = df['Observation']
    model = sm.OLS(Y, X)
    results = model.fit()

    # 输出回归结果
    return results

def plot_regression_results(df, fitted_values, intervention_year, title="Regression Results"):
    """
    绘制回归结果图，包括观察值、拟合值以及干预年份标记。

    :param df: 数据集，包含 "Year" 和 "Observation" 列
    :param fitted_values: 回归模型的拟合值
    :param intervention_year: 干预年份
    :param title: 图表标题，默认为 "Regression Results"
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['Observation'], label='Observed', color='black')  # 观测数据
    plt.plot(df['Year'], fitted_values, label='Fitted', color='red')  # 拟合值
    plt.axvline(x=intervention_year, color='blue', linestyle='--', label='Intervention')  # 干预线
    plt.xlabel('Year')
    plt.ylabel('Observation')
    plt.legend()
    plt.title(f'{title} (Intervention Year: {intervention_year})')
    plt.show()


# 数据集和中断点字典，方便选择
data_sets = {
    "USA_G": (USA_G, intervention_year1),
    "ROU_G": (ROU_G, intervention_year2),
    "CHN_V": (CHN_V, intervention_year3),
    "USA_V": (USA_V, intervention_year4)
}

# 选择数据集名称和对应的干预年份
for name, (df, intervention_year) in data_sets.items():
    print(f"正在处理数据集: {name}，干预年份: {intervention_year}")

    # 分段回归
    results = segmented_regression(df, intervention_year)
    fitted_values = results.fittedvalues

    # 绘制回归结果
    plot_regression_results(df, fitted_values, intervention_year, title=f"{name} Regression Results")

    # 输出回归结果摘
    # print(results.summary())

    # 获取干预系数（β2），即 x2 的系数
    beta_2 = results.params['x2']
    p_value_beta_2 = results.pvalues['x2']
    beta_3 = results.params['x3']
    # 输出 β2 和 p-value
    print(f"β2 (干预的即时效应): {beta_2}")
    print(f"p-value: {p_value_beta_2}")
    print(f"β3 : {beta_3}")
    # 判断干预是否显著
    if p_value_beta_2 < 0.25:
        print("干预的即时效应显著。")

        # 判断干预效果的方向
        if beta_2 > 0:
            print("干预的即时效应是正向的（效果增加）。")
        else:
            print("干预的即时效应是反向的（效果减少）。")
    else:
        print("干预的即时效应不显著。")


# 定义多个国家的数据
countries_data = {
    'India_BDMT': {
        'Year': [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
        'Observation': [0, 0, 0, 0, 0, 1, 2, 1, 0]
    },
    'MEX_BOX': {
        'Year': [1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
        'Observation': [1, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 2]
    },
    'SWE_PP': {
        'Year': [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
        'Observation': [1, 3, 0, 2, 0, 0, 0, 0, 0, 4]
    }
}

# 假设干预年份为2024
intervention_year = 2024

# 循环处理每个国家的数据
for country, data in countries_data.items():
    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 为数据创建干预变量和时间变量
    df['intervention'] = (df['Year'] >= intervention_year).astype(int)
    df['time'] = df['Year'] - df['Year'].min()
    df['post'] = df['Year'] - intervention_year
    df['post'] = df['post'].apply(lambda x: x if x >= 0 else 0)

    # 回归模型设计矩阵
    X = np.column_stack([np.ones(df.shape[0]), df['time'], df['intervention'], df['time'] * df['post']])
    X = sm.add_constant(X)  # 添加常数项

    # 回归分析
    Y = df['Observation']
    model = sm.OLS(Y, X)
    results = model.fit()

    # 获取回归系数 (包括 β1)
    beta_1 = results.params['x1']
    # 假设回归模型的系数（这些系数可以根据实际的回归分析结果得到）
    beta_0 = results.params['const']  # 常数项
    # beta_1 = -0.0326  # 时间变量的系数
    beta_2 = 6.1585  # 干预效应的即时系数
    beta_3 = 0.0050  # 干预效应随时间变化的系数

    # 预测干预前的拟合值
    fitted_values = beta_0 + beta_1 * df['time'] + beta_2 * df['intervention'] + beta_3 * df['time'] * df['post']
    df['Fitted Values'] = fitted_values
    print(fitted_values)
    # 预测从干预年份（2024年）后的数据
    future_years = [2024, 2028, 2032]  # 假设你想预测这些年份
    future_data = pd.DataFrame({'Year': future_years})

    # 为新数据创建干预变量和时间变量
    future_data['intervention'] = (future_data['Year'] >= intervention_year).astype(int)
    future_data['time'] = future_data['Year'] - df['Year'].min()
    future_data['post'] = future_data['Year'] - intervention_year
    future_data['post'] = future_data['post'].apply(lambda x: x if x >= 0 else 0)

    # 使用回归系数预测未来年份的拟合值
    future_data['Fitted Values'] = beta_0 + beta_1 * future_data['time'] + beta_2 * future_data['intervention'] + beta_3 * future_data['time'] * future_data['post']

    # 打印预测结果
    print(f"预测结果 for {country}:")
    print(future_data[['Year', 'Fitted Values']])
    print("\n")