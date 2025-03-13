function [gmm, idx] = fit_gmm(X, K)
    % 高斯混合模型拟合函数
    % 输入：
    %   X - n x m 的数据矩阵，其中n是样本数，m是特征维度
    %   K - 需要拟合的高斯分布成分数
    % 输出：
    %   gmm - 拟合得到的高斯混合模型
    %   idx - 数据点的分类标签

    % 使用MATLAB的fitgmdist函数拟合GMM模型
    gmm = fitgmdist(X, K); % K为高斯成分数
    
    % 使用posterior方法计算每个数据点属于每个高斯成分的概率
    % idx是每个数据点的最大概率成分的标签
    idx = cluster(gmm, X);
    
    % 可视化数据和拟合的GMM模型
    figure;
    scatter(X(:,1), X(:,2), 30, idx, 'filled'); % 根据分类结果绘制不同颜色
    hold on;
    % 绘制GMM的等高线
    ezcontour(@(x,y) pdf(gmm, [x y]), [-5 5], [-5 5], 50); 
    title('高斯混合模型拟合');
    xlabel('X1');
    ylabel('X2');
    colorbar;
end
