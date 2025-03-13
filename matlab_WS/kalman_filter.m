function smoothed_data = kalman_filter(data, Q, R)
    % 卡尔曼滤波降噪
    % 输入:
    %   data: 一维数据（带噪声的观测值）
    %   Q: 过程噪声协方差（预测的信任度）
    %   R: 观测噪声协方差（测量的信任度）
    % 输出:
    %   smoothed_data: 降噪后的数据

    n = length(data);  % 数据长度

    % 初始化卡尔曼滤波器的参数
    x_hat = data(1);    % 初始估计值
    P = 1;              % 初始误差协方差，假设为1
    smoothed_data = zeros(1, n);  % 存储平滑后的数据

    % 卡尔曼滤波
    for k = 1:n
        % 预测步骤
        x_hat_minus = x_hat;           % 预测的状态（此例为恒定模型，因此预测为前一状态）
        P_minus = P + Q;               % 预测的误差协方差

        % 更新步骤
        K = P_minus / (P_minus + R);   % 卡尔曼增益
        x_hat = x_hat_minus + K * (data(k) - x_hat_minus);  % 更新估计值
        P = (1 - K) * P_minus;         % 更新误差协方差

        % 存储当前时刻的平滑值
        smoothed_data(k) = x_hat;
    end
end
