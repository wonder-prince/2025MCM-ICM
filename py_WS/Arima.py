import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
data = pd.read_excel('add_nation.xlsx', header=None)
data = data.transpose()
# 假设你将数据分为训练集和测试集
train, test = train_test_split(data, test_size=0.2, shuffle=False)
data.columns = ['Date', 'Value']
train.columns = ['Date', 'Value']
test.columns = ['Date', 'Value']
# 设置日期列为索引
data.set_index('Date', inplace=True)
# ADF单位根检验，检查数据是否平稳
result = adfuller(data['Value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] < 0.05:
    print("数据是平稳的")
else:
    print("数据是非平稳的，需要差分处理")
# 对数据进行一次差分
data_diff = data['Value'].diff().dropna()

# 重新进行ADF检验
result_diff = adfuller(data_diff)
print('ADF Statistic (差分后):', result_diff[0])
print('p-value (差分后):', result_diff[1])
# 绘制ACF和PACF图
plot_acf(data_diff)
plot_pacf(data_diff)
plt.show()
# 拟合ARIMA模型
model = ARIMA(data['Value'], order=(1, 1, 1))
model_fit = model.fit()

# 查看模型的拟合结果
print(model_fit.summary())

# 确保测试集的大小
test_size = len(test)

# 使用测试集大小作为预测步数
forecast = model_fit.forecast(steps=test_size)
# 打印预测结果
print(forecast)
# 计算误差指标
mae = mean_absolute_error(test['Value'], forecast)
mse = mean_squared_error(test['Value'], forecast)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(test['Value'], forecast)
# 计算准确率
accuracy = 100 - mape * 100

# 打印准确率
print(f"预测准确率: {accuracy:.2f}%")
print(f"MAE (Mean Absolute Error): {mae}")
print(f"MSE (Mean Squared Error): {mse}")
print(f"RMSE (Root Mean Squared Error): {rmse}")
print(f"MAPE (Mean Absolute Percentage Error): {mape}")
