import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc


# 导入数据
total_num = pd.read_excel('total_counts.xlsx')
golden_medal_num = pd.read_excel('Golden_medal_counts.xlsx')
host = pd.read_excel('summerOly_hosts.xlsx')
participants = pd.read_excel('participants.xlsx')
sports = pd.read_excel('sports_num.xlsx')

# 输入参数
nation_name = "China"
nation_mini = "CHN"
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
sports_nation = sports_nation.values //3

# 输出东道主标志
# 初始化一个形状为 (1, 30) 的二维零数组
host_array = np.zeros((1, 30))

# 遍历从1904年到2024年，每4年一次
for year in range(1904, 2025, 4):
    # 查找是否该年份是东道主
    year_host = host[host['Year'] == year]

    # 如果该年是东道主
    if not year_host.empty and year_host['Host'].iloc[0] == nation_name:
        # 计算该年份在数组中的位置
        year_index = (year - 1904) // 4

        # 将对应位置设置为1
        host_array[0, year_index] = 1  # 修改为访问二维数组的特定位置

# 输出 host_array
print(host_array)

# 合并特征
X = np.column_stack(
    [host_array.flatten(), total_nation.flatten(), golden_nation.flatten(), part_num.flatten(), sports_nation.flatten()])

# 创建目标变量（Total nation 是否增加）
y = np.zeros(X.shape[0])
for i in range(1, X.shape[0]):
    if total_nation.flatten()[i] > total_nation.flatten()[i - 1]:
        y[i] = 1  # 如果当前年奖牌数增加，则为1
    else:
        y[i] = 0  # 否则为0

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)
# 预测
y_pred = rf_classifier.predict(X_test)

# 输出分类结果
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 获取各个特征的最后一个值
X_new = pd.DataFrame({
    'Host array': [host_array[0, -1]],  # 获取 Host array 最后一个值
    'Total nation': [total_nation[0,-1]],  # 获取 Total nation 最后一个值
    'Golden nation': [golden_nation[0,-1]],  # 获取 Golden nation 最后一个值
    'Participants number': [part_num[0,-1]],  # 获取 Participants number 最后一个值
    'Sports nation': [sports_nation[0,-1]]  # 获取 Sports nation 最后一个值
})
X_new_array = X_new.values
# 使用模型对新的数据进行预测
#predictions = rf_classifier.predict(X_new)
probabilities = rf_classifier.predict_proba(X_new_array)
# 输出预测结果
#print("Predictions (0: No increase, 1: Increase):", predictions)
print("Predicted probabilities for each class:")
print(probabilities)