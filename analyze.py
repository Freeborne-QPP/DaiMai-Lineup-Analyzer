# 1. 导入工具包
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import feature_synthesis as fs #特征识别文件
import itertools
import random
from tqdm import tqdm
import joblib

df = pd.read_csv('record - 副本.csv',encoding='gbk')  # 若两个文件不在同一文件夹，则需提供完整路径


# 3. 准备特征(X)和目标(y)，假设目标列是'成绩'
X = df.drop(columns=['成绩'])   # DataFrame
y = df['成绩']

plt.hist(y, bins=50)
plt.xlabel('Survived Time')
plt.ylabel('Count')
plt.title('MaiBan S6 SuiZhiyin ShiChang FenBu')
plt.show()

# 4. 划分训练集和测试集（用80%的数据训练，20%的数据验证模型效果）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 创建并训练随机森林模型
# n_estimators=100: 森林里用100棵树
# random_state=42: 固定随机种子，确保每次运行结果一致
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # 把特征X_train和目标y_train喂给模型学习

joblib.dump(model, 'plant_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("模型训练完成，特征顺序已锁定：")
print(X.columns.tolist())

# 6. 评估模型的泛化水平
score = model.score(X_test, y_test)
print(f"R^2 = {score:.4f}")
# R^2分数越接近1，说明模型预测越准

# 7. 查看特征重要性
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n特征重要性排名")
print(importance_df.to_string(index=False))

FEATURE_COLUMNS = ['是否边','保护寒意','保护火焰','保护大C','保护小C','保护对单','保护聚怪','保护真群','保护类机','保护类星','保护经验辅','保护类核','寒意','火焰','硬前排','前排','公式阵','大C','小C','对单','聚怪','真群','类机','类星','经验辅','养嘴','类核','有无狙','有无麦','有无坚','有无雷','有无寒','有无嘴','有无双','有无小','有无阳','有无喷','有无魅','有无胆','有无川','有无窝','有无三','有无缠','有无火','有无高','有无海','有无灯','有无仙','有无叶','有无裂','有无星','有无磁','有无卷','有无玉','有无蒜','有无伞','有无金','有无瓜','有无机','有无曾','有无猫','有无冰','有无吸','有无刺','有无爆','有无飘','有无反','有无若','有无奶','有无幽','有无逆','有无藤']
PLANTS = [
    '狙', '麦', '坚', '雷', '寒', '嘴', '双', '小', '阳', '喷',
    '魅', '胆', '川', '窝', '三', '缠', '火', '高', '海', '灯',
    '仙', '叶', '裂', '星', '磁', '卷', '玉', '蒜', '伞', '金',
    '瓜', '机', '曾', '猫', '冰', '吸', '刺', '爆', '飘', '反',
    '若', '奶', '幽', '逆', '藤'
]
CONSTRAINT_PLANTS = {'坚', '高', '曾', '爆', '仙', '嘴', '玉'}
ALLOWED_POSITIONS = [2, 3, 4]
# 重要：build_features 输出的特征顺序必须与此列表严格一致

# ========== 2. 特征构造函数==========
def build_features(lineup_list):
    """
    输入：阵容列表，如 ['狙','麦','坚','寒','嘴']（长度5）
    输出：包含所有特征值的字典
    """
    # 将列表拼接成字符串，供feature_synthesis使用
    lineup_str = ''.join(lineup_list)
    leihe= fs.count_chars(lineup_str, ['卷', '飘'])
    leihe+= fs.count_chars(lineup_str[:2:], ['胆'])
    count_plant=[0 for i in range(45)]
    for i in range(len(PLANTS)):
        count_plant[i] = fs.count_chars(lineup_str, [PLANTS[i]])
    return {
        '是否边': 0,  # 固定
        '寒意': fs.count_chars(lineup_str, ['寒', '冰', '川']),
        '火焰': fs.count_fire(lineup_str),
        '硬前排': 1 if (fs.contain(lineup_str, ['曾', '爆', '仙', '坚', '高', '嘴']) or
                        fs.dengci(lineup_str,
                                  ['海', '磁', '玉', '蒜', '逆', '伞', '狙', '川', '吸', '奶', '藤'])) else 0,
        '前排': 1 if fs.contain(lineup_str, ['曾', '爆', '仙', '坚', '高', '嘴', '狙', '川', '海', '灯',
                                             '磁', '玉', '蒜', '伞', '吸', '奶', '逆', '藤']) else 0,
        '公式阵': 1 if fs.gongshi(lineup_str) else 0,
        '大C': fs.count_chars(lineup_str, ['雷', '阳', '喷', '瓜', '机', '曾']),
        '小C': fs.count_chars(lineup_str, ['寒', '双', '小', '胆', '窝', '三', '海',
                                           '裂', '星', '卷', '猫', '飘', '反', '若', '奶', '幽']),
        '对单': fs.count_chars(lineup_str, ['寒', '海', '玉', '阳', '缠']),
        '聚怪': fs.count_chars(lineup_str, ['寒', '魅', '川', '缠', '高', '玉', '冰', '刺', '逆']),
        '真群': fs.count_chars(lineup_str, ['雷', '双', '阳', '喷', '窝', '火', '瓜',
                                            '曾', '爆', '飘', '反', '若', '奶', '幽', '藤']),
        '类机': fs.count_chars(lineup_str, ['双', '小', '胆', '裂', '机', '反', '幽']),
        '类星': fs.count_chars(lineup_str, ['阳', '裂', '星', '磁', '曾', '猫', '反', '藤']),
        '经验辅': fs.count_chars(lineup_str, ['麦','三','吸','曾','猫']),
        '类核': leihe,
        '有无狙': count_plant[0],
        '有无麦': count_plant[1],
        '有无坚': count_plant[2],
        '有无雷': count_plant[3],
        '有无寒': count_plant[4],
        '有无嘴': count_plant[5],      # 第6项
        '有无双': count_plant[6],      # 第7项
        '有无小': count_plant[7],      # 第8项
        '有无阳': count_plant[8],      # 第9项
        '有无喷': count_plant[9],      # 第10项
        '有无魅': count_plant[10],     # 第11项
        '有无胆': count_plant[11],     # 第12项
        '有无川': count_plant[12],     # 第13项
        '有无窝': count_plant[13],     # 第14项
        '有无三': count_plant[14],     # 第15项
        '有无缠': count_plant[15],     # 第16项
        '有无火': count_plant[16],     # 第17项
        '有无高': count_plant[17],     # 第18项
        '有无海': count_plant[18],     # 第19项
        '有无灯': count_plant[19],     # 第20项
        '有无仙': count_plant[20],     # 第21项
        '有无叶': count_plant[21],     # 第22项
        '有无裂': count_plant[22],     # 第23项
        '有无星': count_plant[23],     # 第24项
        '有无磁': count_plant[24],     # 第25项
        '有无卷': count_plant[25],     # 第26项
        '有无玉': count_plant[26],     # 第27项
        '有无蒜': count_plant[27],     # 第28项
        '有无伞': count_plant[28],     # 第29项
        '有无金': count_plant[29],     # 第30项
        '有无瓜': count_plant[30],     # 第31项
        '有无机': count_plant[31],     # 第32项
        '有无曾': count_plant[32],     # 第33项
        '有无猫': count_plant[33],     # 第34项
        '有无冰': count_plant[34],     # 第35项
        '有无吸': count_plant[35],     # 第36项
        '有无刺': count_plant[36],     # 第37项
        '有无爆': count_plant[37],     # 第38项
        '有无飘': count_plant[38],     # 第39项
        '有无反': count_plant[39],     # 第40项
        '有无若': count_plant[40],     # 第41项
        '有无奶': count_plant[41],     # 第42项
        '有无幽': count_plant[42],     # 第43项
        '有无逆': count_plant[43],     # 第44项
        '有无藤': count_plant[44]      # 第45项
    }

# ========== 3. 阵容生成器（随机采样/穷举）==========
def generate_random_lineup():
    """随机生成一个符合铲种的阵容（允许重复）"""
    while True:
        lineup = [random.choice(PLANTS) for _ in range(5)]
        # 检查铲种：若前排出现在一号位或二号位，则重新生成
        for i, plant in enumerate(lineup[:2]):  # 只检查索引0,1
            if plant in CONSTRAINT_PLANTS:
                # 出现在禁止位置，重新生成整个阵容
                break
        else:
            return tuple(lineup)

def generate_all_lineups():
    """穷举生成所有符合约束的有序阵容（慎用！50^5≈3.1亿，极慢）"""
    for prod in itertools.product(PLANTS, repeat=5):
        # 检查前两个位置是否有约束植物
        if any(p in CONSTRAINT_PLANTS for p in prod[:2]):
            continue
        yield prod

# ========== 4. 主搜索程序 ==========
def find():
    # ---------- 参数设置 ----------
    MODE = 'random'        # 'random' 或 'exhaustive'
    N_SAMPLES = 3000  # 随机采样数量，根据算力调整
    TOP_K = 35            # 输出前K个阵容

    # ---------- 生成阵容并预测 ----------
    results = []

    if MODE == 'exhaustive':
        print("开始穷举搜索...（可能极慢）")
        generator = generate_all_lineups()
        # 加上进度条（不知道总数，无法显示百分比）
        for lineup in generator:
            features = build_features(lineup)
            # predict要求2D数组，我们构造一行DataFrame
            X_pred = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            score = model.predict(X_pred)[0]
            results.append((lineup, score))
        print(f"穷举完成，共评估 {len(results)} 个阵容")

    else:  # 随机采样
        print(f"开始随机采样，目标数量 {N_SAMPLES} ...")
        for _ in tqdm(range(N_SAMPLES), desc="评估阵容"):
            lineup = generate_random_lineup()
            lineuplist=list(lineup)
            features = build_features(lineuplist)
            X_pred = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            score = model.predict(X_pred)[0]
            results.append((lineup, score))
        print(f"随机采样完成，共评估 {len(results)} 个阵容")

    # ---------- 排序并输出Top K ----------
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n=== AI预测强度前 {TOP_K} 的阵容 ===\n")
    for i, (lineup, score) in enumerate(results[:TOP_K], 1):
        lineup_str = ' '.join(lineup)
        print(f"{i:2d}. 阵容: {lineup_str}  预测成绩 {score:.2f} ")

    # 可选：保存结果到CSV
    # df_out = pd.DataFrame(results, columns=['lineup', 'score'])
    # df_out.to_csv('top_lineups.csv', index=False)


find()
