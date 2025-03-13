% 创建训练数据（XOR问题）
X = [0 0; 0 1; 1 0; 1 1];  % 输入
y = [0; 1; 1; 0];  % 目标输出

% 创建一个前馈神经网络（BP网络），隐藏层有 4 个神经元
hidden_layer_neurons = 4;
net = feedforwardnet(hidden_layer_neurons);

% 设置训练函数（这里使用了默认的 Levenberg-Marquardt 方法）
net.trainFcn = 'trainlm';  % Levenberg-Marquardt 优化算法

% 训练神经网络
[net, tr] = train(net, X', y');

% 使用训练好的网络进行预测
predicted_output = net(X');

% 显示预测结果
disp('Predicted output:');
disp(predicted_output);
