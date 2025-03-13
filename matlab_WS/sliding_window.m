function results = sliding_window(serve, serve_no, p1, p2, L, d)
    % 输入：
    % serve - 发球数据
    % serve_no - 发球次数
    % p1 - 玩家1的相关数据
    % p2 - 玩家2的相关数据
    % L - 时间窗口的长度
    % d - 每次窗口移动的步长

    % 数据长度
    N = length(serve_no);

    % 确保滑动窗口可以进行
    if N < L
        error('数据长度必须大于或等于时间窗口长度');
    end

    % 计算窗口个数
    num_windows = floor((N - L) / d) + 1;

    % 存储每个窗口的处理结果
    results = zeros(2, num_windows);  % 两行分别存储 p1 和 p2 的结果
    
    % 对每个窗口进行处理
    for i = 1:num_windows
        % 初始化变量
        p1_value = 0;
        p2_value = 0;
        p1_count = 0;
        p2_count = 0;
        
        % 计算当前窗口的起始和结束位置
        start_idx = (i - 1) * d + 1;
        end_idx = start_idx + L - 1;
        
        % 提取窗口数据
        window_data = serve_no(start_idx:end_idx);
        p1_data = p1(start_idx:end_idx);
        p2_data = p2(start_idx:end_idx);
        serve_data = serve(start_idx:end_idx);
        
        % 处理窗口数据
        for j = 1:L
            if serve_data(j) == 1
                p1_value = p1_value + p1_data(j) / serve_no(j); % 修正为正确的索引
                p1_count = p1_count + 1;
            elseif serve_data(j) == 2
                p2_value = p2_value + p2_data(j) / serve_no(j); % 修正为正确的索引
                p2_count = p2_count + 1;
            end
        end
        
        % 计算窗口内的平均值
        if p1_count > 0
            results(1, i) = p1_value / p1_count;
        else
            results(1, i) = NaN;  % 如果没有玩家1的数据，返回 NaN
        end
        
        if p2_count > 0
            results(2, i) = p2_value / p2_count;
        else
            results(2, i) = NaN;  % 如果没有玩家2的数据，返回 NaN
        end
    end
    
    % 可选：返回所有窗口的处理结果
    disp('处理结果：');
    disp(results);
end
