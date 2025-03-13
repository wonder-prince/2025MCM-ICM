function [loadings, factors, specificVariances, stats] = performFactorAnalysis(data, numFactors, rotationMethod)
    % 执行因子分析
    % 输入参数：
    %   data - 输入的观测数据，大小为 (n x p)，其中 n 为样本数，p 为变量数
    %   numFactors - 提取的因子数量
    %   rotationMethod - 旋转方法，可选值 'varimax', 'promax', 等
    
    % 输出参数：
    %   loadings - 因子载荷矩阵
    %   factors - 因子得分矩阵
    %   specificVariances - 特有方差
    %   stats - 统计信息，包含拟合优度等
    
    % Step 1: 执行因子分析
    [loadings, factors, specificVariances, stats] = factoran(data, numFactors, 'rotate', rotationMethod);
    
    % Step 2: 显示输出
    disp('因子载荷矩阵：');
    disp(loadings);
    
    disp('因子得分：');
    disp(factors);
    
    disp('特有方差：');
    disp(specificVariances);
    
    disp('因子分析的统计结果：');
    disp(stats);
end
