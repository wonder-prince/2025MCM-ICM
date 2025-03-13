function [idx, C] = kmeans_clustering(X, K)
    % K-means 聚类函数
    % 输入：
    %   X - n x m 的数据矩阵，其中n是样本数，m是特征维度
    %   K - 需要聚类的簇数
    % 输出：
    %   idx - 聚类标签，表示每个点属于哪个簇
    %   C - 聚类中心，K x m 矩阵，其中每一行是一个聚类中心的坐标
    
    % 检查输入数据是否合适
    if size(X, 1) < K
        error('样本数小于簇数，无法进行K-means聚类');
    end

    % 执行K-means聚类
    [idx, C] = kmeans(X, K);
    
    % 可视化数据和聚类结果
    figure;
    gscatter(X(:,1), X(:,2), idx, 'rb', 'xo'); % 绘制不同簇的数据点
    hold on;
    plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3); % 聚类中心
    title('K-means 聚类结果');
    xlabel('X1');
    ylabel('X2');
    axis equal;
    legend('簇1', '簇2', '聚类中心');
end
