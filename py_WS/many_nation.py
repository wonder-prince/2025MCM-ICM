import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 导入数据
def load_data():
    total_num = pd.read_excel('total_counts.xlsx')
    golden_medal_num = pd.read_excel('Golden_medal_counts.xlsx')
    host = pd.read_excel('summerOly_hosts.xlsx')
    participants = pd.read_excel('participants.xlsx')
    sports = pd.read_excel('sports_num.xlsx')

    return total_num, golden_medal_num, host, participants, sports

# 筛选对应国家的数据
def preprocess_data(total_num, golden_medal_num, participants, sports, host, nation_name, nation_mini):
    total_nation = total_num[total_num['NOC'] == nation_name].drop(columns=[total_num.columns[0]])
    golden_nation = golden_medal_num[golden_medal_num['NOC'] == nation_name].drop(columns=[golden_medal_num.columns[0]])
    participants_nation = participants[participants['NOC'] == nation_mini].drop(columns=[participants.columns[0]])
    sports_nation = sports[sports['NOC'] == nation_mini].drop(columns=[sports.columns[0]])

    total_nation = total_nation.values
    golden_nation = golden_nation.values
    part_num = participants_nation.values
    sports_nation = sports_nation.values   # 假设此处为单位转换

    # 输出东道主标志
    host_array = np.zeros((1, 30))

    for year in range(1904, 2025, 4):
        year_host = host[host['Year'] == year]
        if not year_host.empty and year_host['Host'].iloc[0] == nation_name:
            year_index = (year - 1904) // 4
            host_array[0, year_index] = 1

    return total_nation, golden_nation, part_num, sports_nation, host_array

# 训练模型的函数
def train_model(X, y):
    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_model.fit(X, y)
    return rf_model

# 计算模型的性能指标
def evaluate_model(rf_model, X, y):
    y_pred = rf_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

# 进行预测的函数
def predict_new_data(rf_model, host_array, total_nation, golden_nation, part_num, sports_nation):
    X_new = pd.DataFrame({
        'Host array': [host_array[0, -1]],
        'Total nation': [total_nation[0, -1]],
        'Golden nation': [golden_nation[0, -1]],
        'Participants number': [part_num[0, -1]],
        'Sports nation': [sports_nation[0, -1]]
    })

    X_new_array = X_new.values
    prediction = rf_model.predict(X_new_array)
    return prediction, X_new_array

# 计算SHAP值的函数
def calculate_shap_values(rf_model, X):
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

# 可视化SHAP值的函数
def visualize_shap_values(shap_values, X):
    shap.summary_plot(shap_values, X)

def visualize_shap_for_new_data(shap_values_new, explainer, X_new_array):
    shap.force_plot(explainer.expected_value[0], shap_values_new[0], X_new_array[0])

# 主程序
def main():
    nations = [
        {"name": "China", "mini": "CHN"},
        {"name": "United States", "mini": "USA"},
        {"name": "Germany", "mini": "GER"},
        {"name": "Japan", "mini": "JPN"},
        # 可以继续添加更多国家
    ]

    # 导入和处理数据
    total_num, golden_medal_num, host, participants, sports = load_data()
    predictions = []
    for nation in nations:
        nation_name = nation["name"]
        nation_mini = nation["mini"]

        # 导入并处理数据
        total_nation, golden_nation, part_num, sports_nation, host_array = preprocess_data(
            total_num, golden_medal_num, participants, sports, host, nation_name, nation_mini
        )

        # 合并特征
        X = np.column_stack([host_array.flatten(), total_nation.flatten(), golden_nation.flatten(), part_num.flatten(),
                             sports_nation.flatten()])
        y = total_nation.flatten()

        # 训练模型
        rf_model = train_model(X, y)

        # 评估模型
        print(f"Evaluating model for {nation_name}...")
        evaluate_model(rf_model, X, y)

        # 计算SHAP值
        shap_values, explainer = calculate_shap_values(rf_model, X)

        # 可视化SHAP值
        print(f"Visualizing SHAP values for {nation_name}...")
        visualize_shap_values(shap_values, X)

        # 对新的数据进行预测
        prediction, X_new_array = predict_new_data(rf_model, host_array, total_nation, golden_nation, part_num,
                                                   sports_nation)
        predictions.append((nation_name, prediction[0]))  # 保存国家名和对应的预测值
        print(f"Prediction for {nation_name}: {prediction}")

        # 计算新的SHAP值并可视化
        shap_values_new = explainer.shap_values(X_new_array)
        visualize_shap_for_new_data(shap_values_new, explainer, X_new_array)

    # 根据预测值进行排序
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # 输出排序后的结果
    print("\nPredictions sorted by predicted total medals:")
    for nation_name, pred_value in sorted_predictions:
        print(f"{nation_name}: {pred_value}")
# 执行主程序
if __name__ == "__main__":
    main()
