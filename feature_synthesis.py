import pandas as pd
import feature as ft

data=pd.read_csv("C:/Users/asus/AppData/Local/Programs/Python/Python313/PVZBarley/record.csv",encoding='gbk')
print(data.head())
l=len(data)
row=[0 for i in range(l)]
for i in range(l):
    lineup = data.loc[i, "行"]
    if lineup==1:
        row[i] = '1'
    if lineup==5:
        row[i] = '1'
data['是否边']=pd.Series(row)

# 对每一行应用 feature_recog，得到 Series（索引为特征名，值为特征值）
feature_series = data['阵容'].apply(lambda x: pd.Series(ft.feature_recog(x)))

cols_to_overwrite = feature_series.columns.intersection(data.columns)
data = data.drop(columns=cols_to_overwrite)
data = pd.concat([data, feature_series], axis=1)
data.to_csv("C:/Users/asus/AppData/Local/Programs/Python/Python313/PVZBarley/record.csv",encoding='gbk',index=False)