% GM(1,1) 灰色预测模型实现
% 输入: 原始数据 y，输出: 预测数据和模型参数

function [forecast, a, b] = GM11(y)

    % 数据长度
    n = length(y);
    
    % 累加生成
    x1 = cumsum(y);
    
    % 计算数据矩阵 B 和列向量 Y
    B = [-x1(1:n-1), ones(n-1, 1)];
    Y = y(2:n);
    
    % 最小二乘法求解参数 a 和 b
    % a = (B' * B) \ (B' * Y);
    X = inv(B' * B) * B' * Y;
    a = X(1);  % a 系数
    b = X(2);  % b 系数
    
    % 根据模型预测
    forecast = zeros(n, 1);
    forecast(1) = y(1);  % 初始值为原始数据的第一个值
    for k = 2:n
        forecast(k) = (forecast(1) - b / a) * exp(-a * (k - 1));
    end
    
    % 输出模型参数和预测结果
    disp(['模型参数 a: ', num2str(a)]);
    disp(['模型参数 b: ', num2str(b)]);
    disp('预测结果:');
    disp(forecast);
    
end
