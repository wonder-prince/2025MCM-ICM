import shap
import matplotlib.pyplot as plt


def explain_model_with_shap(model, X_train, X_test):
    """
    使用 SHAP 对模型进行解释并绘制 SHAP 值的摘要图和单个样本的解释图。

    参数:
    - model: 已训练好的机器学习模型（例如 RandomForest、XGBoost）
    - X_train: 训练集特征数据
    - X_test: 测试集特征数据
    """
    # 初始化 SHAP 解释器
    if hasattr(model, 'predict_proba'):  # 判断模型是否支持概率预测（分类模型）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:  # 回归模型
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    # 生成 SHAP 值的摘要图
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # 绘制单个样本的 SHAP 解释图
    shap.initjs()  # 可选，初始化 JS 可视化
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0])

    plt.show()

    return shap_values
