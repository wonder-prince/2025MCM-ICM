function feature_importance = measure_association_regression(X, Y, numTrees, feature_names)
    % 使用置换特征重要性方法计算随机森林中的特征与目标变量的关联程度
    % X - 特征矩阵（每行一个样本，列为特征）
    % Y - 目标变量（标签）
    % numTrees - 随机森林中的树木数量
    % feature_names - 变量的名称，长度应与 X 的列数相同
    
    % 创建随机森林模型
    rfModel = TreeBagger(numTrees, X, Y, 'OOBPrediction', 'On', 'Method', 'regression'); % 如果是回归问题
    % rfModel = TreeBagger(numTrees, X, Y, 'OOBPrediction', 'On', 'Method', 'classification'); % 如果是分类问题
    
    % 获取原始的袋外误差（对于回归问题是均方误差，对于分类问题是分类误差）
    oobErrorBefore = oobError(rfModel);
    
    % 如果是回归问题，oobError 返回的是均方误差
    if isnumeric(oobErrorBefore)
        oobErrorBefore = mean(oobErrorBefore); % 直接取均值
    end
    
    % 初始化置换误差
    permutedError = zeros(1, size(X, 2));
    
    for i = 1:size(X, 2)
        % 打乱第 i 个特征的值
        X_permuted = X;
        X_permuted(:, i) = X(randperm(size(X, 1)), i);
        
        % 计算打乱后的袋外误差（选择回归还是分类问题）
        permutedModel = TreeBagger(numTrees, X_permuted, Y, 'OOBPrediction', 'On', 'Method', 'regression'); % 如果是回归问题
        % permutedModel = TreeBagger(numTrees, X_permuted, Y, 'OOBPrediction', 'On', 'Method', 'classification'); % 如果是分类问题
        
        % 获取打乱后模型的袋外误差
        permutedErrorBefore = oobError(permutedModel);
        
        % 如果是回归问题，oobError 返回的是均方误差
        if isnumeric(permutedErrorBefore)
            permutedError(i) = mean(permutedErrorBefore); % 直接取均值
        else
            permutedError(i) = mean(permutedErrorBefore); % 对分类误差取均值
        end
    end
    
    % 计算误差增加量
    errorIncrease = permutedError - oobErrorBefore;
    
    % 输出置换特征重要性
    feature_importance = errorIncrease;
    
    % 输出重要性
    disp('特征重要性：');
    for i = 1:length(feature_names)
        fprintf('%s: %.4f\n', feature_names{i}, feature_importance(i));
    end
    
    % 可视化置换特征重要性
    figure;
    bar(feature_importance);
    set(gca, 'XTickLabel', feature_names); % 设置X轴标签为特征名称
    xlabel('特征');
    ylabel('误差增加');
    title('置换特征重要性');
end
