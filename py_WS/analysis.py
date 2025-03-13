import pandas as pd
import numpy as np
import xgboost as xgb
import shap

# 读取数据
df = pd.read_excel('with_medal.xlsx')
total_num = pd.read_excel('total_counts.xlsx')

# 输入参数
nation_name = "China"
nation_mini = "CHN"

# 获取国家的总数
total_nation = total_num[total_num['NOC'] == nation_name].drop(columns=[total_num.columns[0]])
total_nation = total_nation.values.flatten()  # 将DataFrame转为一维数组

# 获取指定国家的数据
df_nation = df[df['NOC'] == nation_mini]

# 按年份和项目进行分组，并统计每种奖牌的数量
result = df_nation.groupby(['Year', 'Sport']).size().unstack(fill_value=0)

# 显示结果
print("Original result:\n", result)
total_nation_arr = total_nation.flatten()
# 读取 result 的行数
num_rows_result = result.shape[0]
num_rows_total_nation = len(total_nation_arr)

# 如果 total_nation 的行数大于 result 的行数，删除前面的多余行
if num_rows_total_nation > num_rows_result:
    total_nation_arr = total_nation_arr[-num_rows_result:]

# 将 total_nation 转换为 DataFrame
total_nation_df = pd.DataFrame(total_nation_arr, columns=['Total Medals'])

# 将 result 转换为 NumPy 数组（特征数据 X）
X = result.values

# 计算每行的奖牌总数
y = result.sum(axis=1).values  # 每行的总和作为目标变量

# 使用 XGBoost 模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6)
model.fit(X, y)
# 使用 SHAP 来解释模型
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# 获取每个特征的 Shapley 值
shap_values_mean = np.abs(shap_values.values).mean(axis=0)

# 获取对应的运动项目名称
sports = result.columns.tolist()

# 将 Shapley 值与运动项目对应起来
shapley_sports = list(zip(sports, shap_values_mean))

# 根据 Shapley 值排序，获取 Shapley 最大的几个运动项目
top_sports = sorted(shapley_sports, key=lambda x: x[1], reverse=True)

# 输出 Shapley 最大的几个运动项目
top_n = 10  # 输出前5个
print("\nTop {} sports based on Shapley values:".format(top_n))
for sport, value in top_sports[:top_n]:
    print(f"{sport}: Shapley value = {value}")