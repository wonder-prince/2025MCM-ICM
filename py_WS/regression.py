import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
# 导入数据
total_num = pd.read_excel('total_counts.xlsx')
golden_medal_num = pd.read_excel('Golden_medal_counts.xlsx')
host = pd.read_excel('summerOly_hosts.xlsx')
participants = pd.read_excel('participants.xlsx')
sports = pd.read_excel('sports_num.xlsx')

# 输入参数
nation_name = "France"
nation_mini = "FRA"
serve_year = 12

# 筛选对应国家的数据
total_nation = total_num[total_num['NOC'] == nation_name].drop(columns=[total_num.columns[0]])
golden_nation = golden_medal_num[golden_medal_num['NOC'] == nation_name].drop(columns=[golden_medal_num.columns[0]])
participants_nation = participants[participants['NOC'] == nation_mini].drop(columns=[participants.columns[0]])
sports_nation = sports[sports['NOC'] == nation_mini].drop(columns=[sports.columns[0]])

# 转换为 NumPy 数组
total_nation = total_nation.values
golden_nation = golden_nation.values
part_num = participants_nation.values
sports_nation = sports_nation.values

# 输出东道主标志
# 初始化一个形状为 (1, 30) 的二维零数组
host_array = np.zeros((1, 30))

# 遍历从1904年到2024年，每4年一次
for year in range(1904, 2024, 4):
    # 查找是否该年份是东道主
    year_host = host[host['Year'] == year]

    # 如果该年是东道主
    if not year_host.empty and year_host['Host'].iloc[0] == nation_name:
        # 计算该年份在数组中的位置
        year_index = (year - 1904) // 4

        # 将对应位置设置为1
        host_array[0, year_index] = 1  # 修改为访问二维数组的特定位置

# 合并特征
X = np.column_stack(
    [host_array.flatten(), total_nation.flatten(), golden_nation.flatten(), part_num.flatten(), sports_nation.flatten()])

y = total_nation.flatten()  # 假设目标变量是总奖牌数

# 创建并训练模型
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X, y)

# 计算SHAP值
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# 可视化所有特征的SHAP值
shap.summary_plot(shap_values, X)
# 对新的数据进行预测
# 获取各个特征的最后一个值
X_new = pd.DataFrame({
    'Host array': [host_array[0, -1]],  # 获取 Host array 最后一个值
    'Total nation': [total_nation[0,-1]],  # 获取 Total nation 最后一个值
    'Golden nation': [golden_nation[0,-1]],  # 获取 Golden nation 最后一个值
    'Participants number': [part_num[0,-1]],  # 获取 Participants number 最后一个值
    'Sports nation': [sports_nation[0,-1]]  # 获取 Sports nation 最后一个值
})
X_new_array = X_new.values

# 计算该新数据的SHAP值
shap_values_new = explainer.shap_values(X_new_array)
# 预测训练集或测试集的结果
y_pred = rf_model.predict(X)

# 计算 MSE 和 R2
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 输出新的预测SHAP值和预测结果
prediction = rf_model.predict(X_new_array)
print("Prediction for new data:", prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values_new[0], X_new_array[0])