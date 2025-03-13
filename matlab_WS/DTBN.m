function [X_coeff, Y_coeff, X_pred, Y_pred] = DTBN(X, Y,lags)
    % 双时间维度贝叶斯网络（DTBN）模型
    % 输入:
    %   X: 输入的 X_t 时间序列（列向量）
    %   Y: 输入的 Y_t 时间序列（列向量）
    %   lags: 滞后期数，表示使用过去多少个时间点的信息
    % 输出:
    %   X_coeff: X_t的回归系数
    %   Y_coeff: Y_t的回归系数
    %   X_pred: 预测的X_t
    %   Y_pred: 预测的Y_t

    % 确保输入数据是列向量
    if size(X, 2) > size(X, 1)
        X = X';  % 转置为列向量
    end
    if size(Y, 2) > size(Y, 1)
        Y = Y';  % 转置为列向量
    end
    
    T = length(X);  % 时间序列的长度
    
    % 检查数据长度是否一致
    if length(Y) ~= T
        error('X 和 Y 的长度不一致!');
    end

    % 定义贝叶斯网络的结构
    % 假设我们已知X_t和Y_t之间的依赖关系
    X_prior = X(2:end);  % 当前时间点的X
    Y_prior = Y(2:end);  % 当前时间点的Y
    X_past = X(1:end-1);  % 过去时间点的X
    Y_past = Y(1:end-1);  % 过去时间点的Y

    % 线性回归模型来估计参数（近似贝叶斯推断）
    % X_t = a*X_(t-1) + b*Y_(t-1) + error
    % Y_t = c*X_(t-1) + error

    % 对X_t进行回归
    X_coeff = regress(X_prior, [X_past, Y_past]);

    % 对Y_t进行回归
    Y_coeff = regress(Y_prior, X_past);

    % 显示回归系数
    disp('回归系数 X_t = a*X_(t-1) + b*Y_(t-1):');
    disp(X_coeff);  % X_t的回归系数

    disp('回归系数 Y_t = c*X_(t-1):');
    disp(Y_coeff);  % Y_t的回归系数

    % 预测：根据回归系数预测未来的X_t和Y_t
    X_pred = [X_past(end), Y_past(end)] * X_coeff;
    Y_pred = X_past(end) * Y_coeff;

    disp('预测的 X_t 和 Y_t:');
    disp(['X_t: ', num2str(X_pred)]);
    disp(['Y_t: ', num2str(Y_pred)]);
end
