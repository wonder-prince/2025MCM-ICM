function [gprMdl, YPred, YCI] = performGPR(XTrain, YTrain, XTest, kernelFunction, sigma)
    % 高斯过程回归 (GPR) 函数
    % 输入：
    %   XTrain - 训练数据输入（n x p），n为样本数，p为特征数  自变量，可为多个特征
    %   YTrain - 训练数据输出（n x 1），n为样本数             因变量，一个   
    %   XTest  - 测试数据输入（m x p），m为测试样本数  自己设定的
    %   kernelFunction - 核函数类型（例如 'squaredexponential'）
    %   sigma - 噪声标准差
    %
    % 输出：
    %   gprMdl - 训练好的高斯过程回归模型
    %   YPred - 测试数据的预测结果（m x 1）
    %   YCI   - 预测结果的置信区间（m x 2），每行是 [lower bound, upper bound]
    
    % Step 1: 训练高斯过程回归模型
    gprMdl = fitrgp(XTrain, YTrain, 'KernelFunction', kernelFunction, 'Sigma', sigma);

    % Step 2: 使用训练好的模型进行预测
    [YPred, YCI] = predict(gprMdl, XTest);
    
    % 显示模型和预测结果
    disp('训练好的高斯过程回归模型：');
    disp(gprMdl);
    
    disp('预测结果：');
    disp(YPred);
    
    disp('预测的置信区间：');
    disp(YCI);
end
