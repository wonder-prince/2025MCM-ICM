% 手动输入数据
data = [11, 10, 0, 6, 0, 4, 5, 6, 0, 4, 5, 6, 0, 6, 0, 3, 4, 0, 0, 11, 5, 12, 15, 1, 0, 13, 0, 0, 7, 0];

[fitModel, forecast, forecastMSE, residuals] = fit_arima_forecast(data', 0, 1, 1, 3);
disp(residuals)
disp(forecast)
% 计算实际值和预测值的误差
% 假设我们把最后的几个数据点作为测试集进行比较（例如，最后3个点是我们预测的目标）
actualValues = data(end-2:end);  % 真实值（最后3个数据点）
forecastedValues = forecast;  % 预测值（对应的3个点）

% 计算每个预测的百分比误差
absolutePercentageError = abs((actualValues - forecastedValues) ./ actualValues) * 100;

% 计算 MAP (Mean Absolute Percentage Error)
mape = mean(absolutePercentageError);

% 输出 MAPE 值
disp(['MAPE: ', num2str(mape), '%']);