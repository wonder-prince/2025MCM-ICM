function shap_values = train_and_compute_shap(X_train, y_train, X_test)
    % train_and_compute_shap: 训练模型并计算SHAP值
    %
    % 输入:
    %   X_train - 训练数据，矩阵，大小为 [n_samples, n_features]
    %   y_train - 训练标签，矩阵，大小为 [n_samples, 1]
    %   X_test  - 测试数据，矩阵，大小为 [m_samples, n_features]
    %
    % 输出:
    %   shap_values - 计算出的SHAP值，数组

    % 设置Python环境
    pyenv('Version', 'D:\conda_data\envs\python311\python.exe');  % 修改为您的Python路径

    % 导入所需的Python库
    xgboost = py.importlib.import_module('xgboost');
    shap = py.importlib.import_module('shap');

    % 将MATLAB数据转换为Python格式
    X_train_py = py.numpy.array(X_train);
    y_train_py = py.numpy.array(y_train);
    X_test_py = py.numpy.array(X_test);

    % 训练XGBoost模型
    dtrain = xgboost.DMatrix(X_train_py, pyargs('label', y_train_py));  % 数据转换为XGBoost格式
    params = py.dict(pyargs('objective', 'reg:squarederror'));  % 回归任务
    model = xgboost.train(params, dtrain, int32(100));  % 训练模型，100轮

    % 计算SHAP值
    explainer = shap.TreeExplainer(model);  % 创建解释器
    shap_values_py = explainer.shap_values(X_test_py);  % 计算SHAP值

    % 将Python中的SHAP值转换为MATLAB格式
    shap_values = double(py.array.array('d', shap_values_py));

    % 返回SHAP值
end
