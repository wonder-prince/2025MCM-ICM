function smoothed_data = gaussian_filter(data, sigma)
    % 高斯滤波函数
    % 输入:
    %   data: 原始数据，可以是1D或2D矩阵
    %   sigma: 高斯滤波器的标准差
    % 输出:
    %   smoothed_data: 滤波后的数据

    % 判断输入数据是1D还是2D
    if isvector(data)
        % 对一维数据进行高斯滤波
        kernel_size = 2 * ceil(3 * sigma) + 1;  % 设置高斯核的大小
        x = -floor(kernel_size/2):floor(kernel_size/2);
        gaussian_kernel = exp(-(x.^2) / (2 * sigma^2));
        gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);  % 归一化

        % 扩展数据：将数据两端重复扩展
        padding = floor(kernel_size / 2);
        extended_data = [repmat(data(1), 1, padding), data, repmat(data(end), 1, padding)];

        % 对扩展后的数据进行卷积，使用'same'来保留与原数据相同的大小
        smoothed_data = conv(extended_data, gaussian_kernel, 'valid');
         % 绘制原始数据和平滑后数据
        figure;
        hold on;
        plot(data, 'b', 'LineWidth', 2);  % 原始数据用蓝色
        plot(smoothed_data, 'r', 'LineWidth', 2);  % 平滑后数据用红色
        legend('原始数据', '平滑后数据');
        title('原始数据与平滑后数据对比');
        xlabel('数据点');
        ylabel('值');
        grid on;
        hold off;
    elseif ismatrix(data)
        % 对二维数据进行高斯滤波
        kernel_size = 2 * ceil(3 * sigma) + 1;  % 设置高斯核的大小
        [x, y] = meshgrid(-floor(kernel_size/2):floor(kernel_size/2), ...
                          -floor(kernel_size/2):floor(kernel_size/2));
        gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
        gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));  % 归一化

        % 对二维数据进行卷积，使用'replicate'边界方式
        smoothed_data = imfilter(data, gaussian_kernel, 'replicate');
    else
        error('输入数据必须是向量或矩阵');
    end
end
