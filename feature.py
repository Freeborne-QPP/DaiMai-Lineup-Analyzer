import pandas as pd

def remove_spaces(input_str):
    chars = input_str.split(" ")
    result = "".join(chars)
    return result


def contain(input_str, char_list):
    for i in range(2,len(input_str)):
        if input_str[i] in char_list:
            return True
    return False

def check_dengci(input_str, char_list):
    if "刺" not in input_str:
        return False

    ci_index = input_str.index("刺")

    if ci_index > 0:
        prev_char = input_str[ci_index - 1]
        if prev_char in char_list:
            return True

    if ci_index > 1:
        prev_prev_char = input_str[ci_index - 2]
        if prev_prev_char == "灯":
            return True

    return False

def count_chars(s, char_list):
    return sum(1 for char in s if char in char_list)

data=pd.read_csv("C:/Users/asus/AppData/Local/Programs/Python/Python313/PVZBarley/record.csv",encoding='gbk')
print(data.head())
l=len(data)
feature=['0' for _ in range(l)]
for i in range(l):
    lineup = data.loc[i, "成绩"]
    feature[i]=1 if lineup>120000 else 0
data['强阵']=pd.Series(feature)
data.to_csv("C:/Users/asus/AppData/Local/Programs/Python/Python313/PVZBarley/record.csv",encoding='gbk',index=False)