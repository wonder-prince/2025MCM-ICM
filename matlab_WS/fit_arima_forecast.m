function [fitModel, forecastValues, forecastMSE, residuals] = fit_arima_forecast(y, p, d, q, numForecasts)
    % fit_arima_forecast: 用于拟合 ARIMA 模型并进行预测
    % 输入：
    %   y - 时间序列数据（列向量）
    %   p - 自回归阶数
    %   d - 差分阶数
    %   q - 滑动平均阶数
    %   numForecasts - 预测未来的时间步数
    % 输出：
    %   fitModel - 拟合的 ARIMA 模型
    %   forecastValues - 预测结果
    %   forecastMSE - 预测的均方误差
    %   residuals - 模型的残差

    % Step 1: Fit the ARIMA model
    model = arima(p, d, q);  % ARIMA(p, d, q)
    
    % 估计模型参数
    fitModel = estimate(model, y);
    
    % Step 2: Check residuals (model diagnostics)
    residuals = infer(fitModel, y);  % 计算模型的残差
    
    % Step 3: Forecast the next 'numForecasts' data points
    % 使用拟合好的模型进行预测
    [forecastValues, forecastMSE] = forecast(fitModel, numForecasts, 'Y0', y);  % 只传递 fitModel 和 numForecasts
    
    % Step 4: Plot the results
    figure;
    hold on;
    plot(y, 'b', 'LineWidth', 2);  % Plot the original data
    plot(length(y) + (1:numForecasts), forecastValues, 'r', 'LineWidth', 2);  % Plot the forecast
    fill([length(y) + (1:numForecasts), length(y) + (1:numForecasts)] - 1, ...
        [forecastValues' - 1.96*sqrt(forecastMSE)', forecastValues' + 1.96*sqrt(forecastMSE)'], ...
        'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  % Confidence interval
    title('ARIMA Forecast');
    xlabel('Time');
    ylabel('Value');
    legend('Original Data', 'Forecasted Data', '95% Confidence Interval');
    hold off;
    
    % Step 5: Plot residuals and autocorrelation of residuals
    figure;
    subplot(2,1,1);
    plot(residuals);
    title('Residuals');
    subplot(2,1,2);
    autocorr(residuals);
    title('ACF of Residuals');
end
