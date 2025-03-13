function [Q, pValue] = ljungBoxTest(residuals, lags)
    % Ljung-Box Q 检验
    % residuals: 模型残差序列
    % lags: 检验的滞后阶数
    
    % 获取残差的长度
    n = length(residuals);
    
    % 计算残差的自相关函数（ACF）
    autocorr = xcorr(residuals, lags, 'coeff');
    autocorr = autocorr(lags+1:end);  % 取滞后阶数以内的自相关系数
    
    % 计算 Ljung-Box Q 统计量
    denominator = n - (1:lags);  % 计算分母
    Q = n * (n + 2) * sum(autocorr.^2 ./ denominator);
    
    % 计算 p-value
    pValue = 1 - chi2cdf(Q, lags);
    
    % 显示结果
    fprintf('Ljung-Box Q-statistic: %.4f\n', Q);
    fprintf('p-value: %.4f\n', pValue);
end
