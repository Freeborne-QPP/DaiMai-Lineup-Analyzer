# 编写过程借助DeepSeek V3.2
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


# ==================== 配置部分 ====================
DATA_PATH = 'record.csv'                # 训练数据文件路径
TARGET_COL = '成绩'               # 目标列名
TEST_SIZE = 0.2                             # 测试集比例
RANDOM_STATE = 42                            # 随机种子
MODEL_SAVE_PATH = 'gb_model.pkl'             # 模型保存路径
FEATURE_SAVE_PATH = 'gb_feature_columns.pkl' # 特征列名保存路径
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 梯度提升模型参数（可按需调整）
GB_PARAMS = {

    'n_estimators': 1000,               # 增加到足够大的值
    'learning_rate': 0.1,
    'max_depth': 4,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'subsample': 0.8,
    'validation_fraction': 0.1,          # 将训练集的 10% 作为验证集
    'n_iter_no_change': 10,               # 连续 10 次迭代验证集损失无改善则停止
    'tol': 1e-4,                          # 改善的容忍阈值
    'random_state': RANDOM_STATE,
    'verbose': 1                    # 训练时打印过程
}
# =================================================

# 1. 加载数据
df = pd.read_csv(DATA_PATH,encoding='gbk')
print(f"数据加载成功，共 {df.shape[0]} 行，{df.shape[1]} 列")

# 2. 分离特征和目标
print("实际列名：", df.columns.tolist())
print("目标列名变量 TARGET_COL =", TARGET_COL)
y = df[TARGET_COL]
X_raw = df.drop(columns=['行','阵容','成绩'])

print(X_raw.columns)
# 3. 特征类型识别与独热编码
#   假设数值特征列名为 numerical_features，分类特征列名为 categorical_features
#   请根据你的实际列名修改这两个列表！
numerical_features = ['硬前排', '前排', '寒意', '火焰', '大C', '小C', '经验量', '对单',
       '聚怪', '真群', '类机', '类星', '经验辅', '养嘴', '类核', '中期C', '最后输出', '最前防御',
        '保护_寒意', '保护_火焰', '保护_大C', '保护_小C', '保护_对单',
       '保护_聚怪', '保护_真群', '保护_类机', '保护_类星', '保护_经验辅', '保护_类核', '保护_中期C', '狙数',
       '麦数', '坚数', '雷数', '寒数', '嘴数', '双数', '小数', '阳数', '喷数', '魅数', '川数', '三数',
       '缠数', '火数', '高数', '海数', '灯数', '仙数', '叶数', '裂数', '星数', '磁数', '卷数', '玉数',
       '蒜数', '伞数', '金数', '瓜数', '机数', '曾数', '猫数', '冰数', '吸数', '刺数', '爆数', '飘数',
       '反数', '若数', '奶数', '幽数', '逆数', '藤数', '前窝', '后胆'] 
categorical_features = ['公式阵','是否边','一号抗', '二号抗', '三号抗','四号抗']  # ,'plant_1','plant_2','plant_3','plant_4','plant_5'

# 确保列名存在于数据中
available_num = [col for col in numerical_features if col in X_raw.columns]
available_cat = [col for col in categorical_features if col in X_raw.columns]
print(f"实际使用的数值特征: {available_num}")
print(f"实际使用的分类特征: {available_cat}")

# 对分类特征进行独热编码
if available_cat:
    X_encoded = pd.get_dummies(X_raw[available_cat], prefix_sep='_')
    # 将编码后的分类特征与数值特征合并
    X = pd.concat([X_raw[available_num], X_encoded], axis=1)
else:
    X = X_raw[available_num].copy()

# 数值特征用中位数填充
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X = X_imputed

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 5. 训练梯度提升模型
model = GradientBoostingRegressor(**GB_PARAMS)
model.fit(X_train, y_train)
print("模型训练完成")

# 6. 评估模型
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n=== 模型评估 ===")
print(f"训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
print(f"测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# 7. 特征重要性分析
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n=== 特征重要性排名 (Top 15) ===")
print(importance_df.head(15).to_string(index=False))

# 可选：绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances (Gradient Boosting)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 8. 保存模型和特征列名
joblib.dump(model, MODEL_SAVE_PATH)
joblib.dump(X.columns.tolist(), FEATURE_SAVE_PATH)
print(f"\n模型已保存至 {MODEL_SAVE_PATH}")
print(f"特征列名已保存至 {FEATURE_SAVE_PATH}")
