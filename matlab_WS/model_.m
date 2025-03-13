T = readtable('data_dictionary.csv');
S = readtable('Wimbledon_featured_matches.csv');
index = 30;
% 获取 score 列并转换为数字
score = S{1:index, 16};  % 只取前 3 行，第 16 列
game = S{1:index, 19};  % 只取前 3 行，第 19 列
serve = S{1:index, 14};  % 发球
serve_no = S{1:index, 15}; %发球次数
points = S{1:index, 17}; %得分
several_num = S{1:index ,13:19};
%两个选手的得分
p1_points_won = S{1:index, 12};
p2_points_won = S{1:index, 13};
pointdif= p1_points_won - p2_points_won;    %得分差异
test_num = [1,13,10,1];
%发球的次数
results = sliding_window(serve, serve_no, p1_points_won, p2_points_won, 13, 5);
n = length(results);
M = zeros(2,n);
for i = 1:n
    P = [results(1,i), n - results(2,i); n - results(1,i), results(2,i)];
    M(:,i) = P*[0.5;0.5];
end
disp(M);
% ljungBox检验方法
[Q, pValue] = ljungBoxTest(p1_points_won, 10);
% 双时间维度贝叶斯网络模型（验证关系和预测）[回归系数阈值：0.05]
[X_coeff, Y_coeff, X_pred, Y_pred] = DTBN(p1_points_won, serve_no, 1);
% 使用随机森林回归模型衡量两个变量的关联程度
feature_importance = measure_association_regression(p1_points_won, serve_no, 100);
% 使用高斯滤波对一维和二维的数据进行平滑处理
smoothed_data = gaussian_filter(Q, 0.5);
% 使用卡尔曼滤波对一维数据进行处理
kalman_data = kalman_filter(Q, 0.001, 0.1);
% 使用ARIMA时间序列模型 (使用（1，1，1）时间序列）[可以使用ljugBox来检验残差，从而评估模型的拟合程度]
[fitModel, forecast, forecastMSE, residuals] = fit_arima_forecast(points, 1, 1, 1, 3);
% 使用因子分析法对多个变量进行降维
%[loadings, factors, specificVariances, stats] = performFactorAnalysis(several_num, 2,'varimax');
% 使用高斯过程回归（GRP）进行模型训练与预测
%[gprMdl, YPred, YCI] = performGPR(several_num, serve, test_num, 'squaredexponential', 0.1);
% 使用k聚类对数据进行相似分类
%[idx, C] = kmeans_clustering(several_num, 2);
% 使用高斯混合模型对数据进行分类
%[gmm, idx_gmm] = fit_gmm(several_num, 3);
%计算变异系数
%cv = calculate_cv(data);l
% 计算位置熵
% 定义任意长度的概率向量
probabilities = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1];
% 调用函数计算总位置熵
total_entropy = compute_position_entropy(probabilities);
% 输出结果
disp(['Total Position Entropy: ', num2str(total_entropy)]);
% 灰色预测
[forecast, a, b] = GM11(points);
% 绘制原始数据与预测数据
figure;
plot(1:length(points), points, 'b-o', 'DisplayName', '原始数据');
hold on;
plot(1:length(forecast), forecast, 'r--', 'DisplayName', '预测数据');
legend;
xlabel('时间');
ylabel('数值');
title('GM(1,1) 灰色预测');
% %disp(score);
% outliers = isoutlier(score);
% disp(outliers);
% c_a = 0;
% c_b = 0;
% g_a = 0;
% g_b = 0;
% motion_1 = 0;
% motion_2 = 0;
% for i = 1:index
% end
% 
% for i = 1:index
%     score_win = score(i);
% 
%    if game(i) == 1
%         g_a  = g_a+1;
%         c_a = c_a + 1;
%    elseif game(i) == 2
%         g_b  = g_b+1;
%         c_b = c_b + 1;
%    else    
%     if score_win == 1
%         c_a = c_a + 1;
%     elseif score_win == 2
%         c_b = c_b + 1;
%     end
%     end
% end

% disp(['Player A wins: ', num2str(c_a)]);
% disp(['Player B wins: ', num2str(c_b)]);
% disp(['Player A wins: ', num2str(g_a)]);
% disp(['Player B wins: ', num2str(g_b)]);
