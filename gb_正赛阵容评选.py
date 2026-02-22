import pandas as pd
import joblib
import numpy as np
import feature as ft  # 你的特征提取模块，确保它能返回与训练时完全一致的特征（包括独热编码后的列）

# ==================== 配置部分 ====================
# 修改点：模型文件名改为梯度提升模型的保存文件
MODEL_PATH = 'gb_model.pkl'               # 原为 'plant_model.pkl'
FEATURE_PATH = 'gb_feature_columns.pkl'    # 原为 'feature_columns.pkl'
# =================================================

def lineup_to_features(lineup_str):
    features = ft.feature_recog(lineup_str)
    return features

def main():
    # 1. 加载训练好的模型和特征列
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURE_PATH)
        print(f"模型加载成功！依赖 {len(feature_columns)} 个特征")
    except FileNotFoundError as e:
        print(f"错误：找不到模型文件，请确保 {MODEL_PATH} 和 {FEATURE_PATH} 在当前目录下")
        return

    # 2. 读取 mcdm.txt 文件
    try:
        with open('mcdm.txt', 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("错误：找不到 mcdm.txt 文件")
        return

    # 3. 筛选长度为5的行作为阵容
    lineups = [line for line in lines if len(line) == 5]
    print(f"共找到 {len(lineups)} 个有效阵容")

    if not lineups:
        print("没有找到任何长度为5的阵容")
        return

    # 4. 对每个阵容进行预测
    results = []
    for lineup in lineups:
        # 计算特征（确保 feat_dict 包含所有需要的特征列，包括独热编码后的列）
        feat_dict = lineup_to_features(lineup)
        # 转换为DataFrame，并确保特征列与训练时一致
        df = pd.DataFrame([feat_dict])
        # 对齐特征列（缺失的填0，多余的删除）
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_columns]

        # 预测（梯度提升模型与随机森林的predict接口一致）
        pred = model.predict(X)[0]
        results.append((lineup, pred))

    # 5. 按预测值降序排序，取前30（可根据需要调整数量）
    results.sort(key=lambda x: x[1], reverse=True)
    top30 = results[:30]  # 原为100，这里改为30，你可自行调整

    # 6. 输出结果
    print("\nAI预测正赛阵容排行")
    print(f"{'排名':<4} {'阵容':<10} {'预测成绩':<10}")
    for i, (lineup, pred) in enumerate(top30, 1):
        print(f"{i:<4} {lineup:<10} {pred:<10.2f}")

    # 可选：将结果保存到文件
    with open('top30_lineups.txt', 'w', encoding='utf-8') as f:
        f.write("排名\t阵容\t预测成绩\n")
        for i, (lineup, pred) in enumerate(top30, 1):
            f.write(f"{i}\t{lineup}\t{pred:.2f}\n")
    print("\n结果已保存到 top30_lineups.txt")

if __name__ == "__main__":
    main()