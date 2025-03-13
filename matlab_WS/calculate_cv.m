function cv = calculate_cv(data)
    % 计算一组数据的变异系数（Coefficient of Variation, CV）
    % 输入：
    %   data - 一个包含数据的向量或矩阵
    % 输出：
    %   cv - 变异系数，单位：百分比（%）

    % 确保输入数据为数值型
    if ~isnumeric(data)
        error('输入数据必须为数值型');
    end
    
    % 计算均值和标准差
    mean_val = mean(data);  % 均值
    std_val = std(data);    % 标准差
    
    % 计算变异系数
    if mean_val ~= 0
        cv = (std_val / mean_val) * 100;  % 变异系数，百分比表示
    else
        cv = NaN;  % 如果均值为零，变异系数无法定义，返回NaN
        disp('警告：数据的均值为零，无法计算变异系数');
    end
end
