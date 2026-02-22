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
def count_fire(s):

    # 检查长度
    if len(s) != 5:
        raise ValueError("输入字符串长度必须为5")

    # 1. 检查是否包含"火"
    if "火" not in s:
        return 0

    # 2. 找到最右边的"火"的位置
    rightmost_fire = s.rindex("火")

    count = 0

    # 3. 计数左边的元素
    left_elements = {"机", "三", "裂", "双", "反"}
    for i in range(rightmost_fire):
        if s[i] in left_elements:
            count += 1

    # 4. 计数右边的元素
    right_elements = {"裂", "反"}
    for i in range(rightmost_fire + 1, len(s)):
        if s[i] in right_elements:
            count += 1

    return count##
def dengci(s, ruanqianpai):
    """
    检查字符串S中最右边的"刺"左边是否包含列表A中的任意元素

    参数:
        s: 长度为5的字符串
        A: 字符列表

    返回:
        bool: 符合条件返回True，否则返回False
    """
    # 检查长度
    if len(s) != 5:
        raise ValueError("输入字符串长度必须为5")

    # 1. 检查是否包含"刺"
    if "刺" not in s:
        return False

    # 2. 找到最右边的"刺"的位置
    rightmost_ci = s.rindex("刺")

    # 3. 如果"刺"在第一个位置，左边没有元素
    if rightmost_ci == 0:
        return False

    # 4. 检查左边是否包含列表A中的元素
    left_part = s[:rightmost_ci]
    for char in left_part:
        if char in ruanqianpai:
            return True

    return False
def gongshi(s):

    # 定义类别字符
    A_chars = {'雷', '双', '小', '阳', '喷', '胆', '窝', '三', '裂', '星',
               '卷', '瓜', '机', '曾', '飘', '反', '若', '奶', '幽'}
    B_chars = {'寒', '玉', '阳', '海', '缠'}
    C_chars = {'坚', '高', '曾', '嘴', '爆', '仙'}  # 虽然没用到，但保留定义
    D_chars = {'坚', '高', '曾', '嘴', '爆', '仙', '狙', '麦', '川', '海',
               '灯', '磁', '玉', '蒜', '伞', '吸', '若', '奶', '逆', '藤'}

    # 条件1: 字符串里有D类字符，特殊处理"灯"
    temp_d_positions = set()
    d_positions = set()

    # 首先找出所有的D类字符位置
    for i, char in enumerate(s):
        if char in D_chars:
            d_positions.add(i)

    # 特殊处理"灯"右边的字符
    for i in range(len(s)):
        if s[i] == '灯' and i + 1 < len(s):
            temp_d_positions.add(i + 1)

    all_d_positions = d_positions.union(temp_d_positions)

    # 条件1判定
    if not all_d_positions:
        return False

    # 条件2: 字符串里有A类字符，且至少有一个A类字符在D类字符左边
    condition2_met = False
    for i, char in enumerate(s):
        if char in A_chars:
            # 检查是否有D类字符在它的右边（包括自身是D类字符的情况）
            for d_pos in all_d_positions:
                if d_pos >= i:  # 在右边或同一位置
                    condition2_met = True
                    break
        if condition2_met:
            break

    if not condition2_met:
        return False

    # 条件3: 存在"缠"或B类字符在D类字符左边
    condition3_met = False

    # 检查是否有"缠"
    if '缠' in s:
        condition3_met = True
    else:
        # 检查B类字符是否在D类字符左边（包括自身是D类字符的情况）
        for i, char in enumerate(s):
            if char in B_chars:
                for d_pos in all_d_positions:
                    if d_pos >= i:  # 在右边或同一位置
                        condition3_met = True
                        break
            if condition3_met:
                break

    return condition3_met
def baohu(str,char_list=['曾', '爆', '仙', '坚', '高', '嘴','狙','川','海','灯','磁','玉','蒜','伞','吸','奶','逆','藤']):
    s=list(str)
    for i in [4,3,2,1,0]:
            if s[i] in char_list:
                break
            else:
                if not s[i-1]=='灯':
                    s[i]='空'
                else:
                    break
    return ''.join(s),i+1
def ronghua(lineup):
    if not '藤' in lineup:
        return 0
    else:
        fire=0
        ice=0
        rate=1
        for i in range(5):
            if lineup[i]=='寒':
                ice+=1.5
            if lineup[i]=='冰':
                ice+=3
            if lineup[i]=='川':
                rate*=2
            if lineup[i]=='飘':
                fire+=1.5
        fire+=count_fire(lineup)*1.5
        ice*=rate
        return min(ice,fire)
def kangxing(lineup):
    kang1=kang2=kang3=kang4=0
    if lineup[0] in ['狙','川','火','三','海','灯','伞','冰','吸','奶','逆','藤'] or lineup[1] in ['伞','灯']:
        kang1=1
    if lineup[1] in ['狙','麦','魅','川','火','海','灯','磁','伞','冰','吸','奶','逆','藤'] or lineup[2] in ['伞','灯'] or lineup[0]=='灯':
        kang2=1
    if lineup[2] in ['狙', '麦', '魅', '川', '火', '海', '灯', '磁', '伞', '冰', '吸', '奶', '逆', '藤'] or lineup[3] in ['伞', '灯'] or lineup[1]=='灯':
        kang3 = 1
    if lineup[3] in ['狙', '麦', '魅', '川', '火', '海', '灯', '磁', '伞', '冰', '吸', '奶', '逆', '藤'] or lineup[4] in ['伞', '灯'] or lineup[2]=='灯':
        kang4 = 1
    return kang1,kang2,kang3,kang4
def count_exp(lineup):
    exp_list={'狙':7500,'麦':0,'坚':3000,'雷':100000,'寒':45000,'嘴':22500,'双':45000,'小':45000,'阳':125000,
'喷':100000,'魅':30000,'胆':45000,'川':15000,'窝':45000,'三':45000,'缠':15000,'火':30000,'高':3000,
'海':45000,'灯':11250,'仙':3000,'叶':11250,'裂':45000,'星':45000,'磁':7500,'卷':45000,'玉':15000,
'蒜':3000,'伞':7500,'金':11250,'瓜':100000,'机':100000,'曾':100000,'猫':45000,'冰':15000,'吸':11250,
'刺':3000, '爆':3000,'飘':45000,'反':45000,'若':45000,'奶':45000,'幽':45000,'逆':7500,'藤':11250}
    exp=0
    for i in range(5):
        if lineup[i] in exp_list:
            exp+=exp_list[lineup[i]]
    return exp

def feature_recog(lineup):
    feature = {}

    if contain(lineup, ['曾', '爆', '仙', '坚', '高', '嘴']):
        feature['硬前排'] = count_chars(lineup, ['曾', '爆', '仙', '坚', '高', '嘴'])
    else:
        if dengci(lineup, ['海', '磁', '玉', '蒜', '逆', '伞', '狙', '川', '吸', '奶', '藤']):
            feature['硬前排'] = 1
        else:
            feature['硬前排']=0

    feature['公式阵'] = 1 if gongshi(lineup) else 0
    feature['前排'] = 1 if contain(lineup, ['曾', '爆', '仙', '坚', '高', '嘴','狙','川','海','灯','磁','玉','蒜','伞','吸','奶','逆','藤']) else 0
    feature['寒意'] = count_chars(lineup,['寒','冰','川'])
    feature['火焰'] = count_fire(lineup)
    feature['大C']=count_chars(lineup,['雷','阳','喷','瓜','机','曾'])
    feature['小C']=count_chars(lineup,['寒','双','小','三','裂','星','卷','飘','反','若','奶','幽'])#删海猫胆窝
    feature['小C']+=count_chars(lineup[:2:],['胆'])
    feature['小C']+=count_chars(lineup[2::],['窝'])
    feature['经验量']=count_exp(lineup)
    feature['对单']=count_chars(lineup,['寒','海','玉','阳','缠'])
    feature['聚怪']=count_chars(lineup,['寒','魅','川','缠','高','玉','冰','刺','逆'])
    feature['真群']=count_chars(lineup,['雷','双','阳','喷','窝','火','瓜','曾','爆','飘','反','若','奶','幽','藤'])
    feature['类机']=count_chars(lineup[:3:],['双','小','胆','裂','机','幽'])+count_chars(lineup,['反'])
    feature['类星']=count_chars(lineup[2::],['阳','裂','星','磁','曾','猫','反','藤'])
    feature['经验辅']=count_chars(lineup,['麦','三','吸','曾','猫'])
    feature['养嘴']=count_chars(lineup[3::],['嘴'])
    a=count_chars(lineup,['卷','飘'])
    a+=count_chars(lineup[:2:],['胆'])
    feature['类核']=a
    feature['中期C']=count_chars(lineup,['机','寒','双','三','裂','星','飘','反','若','奶','幽'])+count_chars(lineup[1::],['小'])+count_chars(lineup[2::],['窝'])
    #feature['融化加成']=ronghua(lineup)
    feature['一号抗'],feature['二号抗'],feature['三号抗'],feature['四号抗']=kangxing(lineup)
    feature['最后输出'] = 4
    if '反' in lineup:
        feature['最后输出']=1
    else:
        for i in range(5):
            if lineup[i] in ['雷','双','小','阳','喷','胆','窝','三','海','裂','卷','瓜','机','冰','飘','若','奶','幽']:
                feature['最后输出']=i+1
                break
    linewq,feature['最前防御']=baohu(lineup) #无前排
    feature['保护_寒意'] = count_chars(linewq,['寒','冰','川'])
    feature['保护_火焰'] = count_fire(linewq)
    feature['保护_大C'] = count_chars(linewq,['雷','阳','喷','瓜','机','曾'])
    feature['保护_小C'] = count_chars(linewq,['寒','双','小','胆','窝','三','裂','星','卷','飘','反','若','奶','幽'])
    feature['保护_对单'] = count_chars(linewq,['寒','海','玉','阳','缠'])
    feature['保护_聚怪'] = count_chars(linewq,['寒','魅','川','缠','高','玉','冰','刺','逆'])
    feature['保护_真群'] = count_chars(linewq,['雷','双','阳','喷','窝','火','瓜','曾','爆','飘','反','若','奶','幽','藤'])
    feature['保护_类机'] = count_chars(linewq[:3:],['双','小','胆','裂','机','幽'])+count_chars(lineup,['反'])
    feature['保护_类星'] = count_chars(linewq[2::],['阳','裂','星','磁','曾','猫','反','藤'])
    feature['保护_经验辅'] = count_chars(linewq,['麦','三','吸','曾','猫'])
    a = count_chars(linewq,['卷','飘'])
    a += count_chars(linewq[:2:],['胆'])
    feature['保护_类核'] = a
    feature['保护_中期C'] = count_chars(linewq,
                                   ['机', '寒', '双', '三', '裂', '星', '飘', '反', '若', '奶', '幽']) + count_chars(
        lineup[1::], ['小']) + count_chars(lineup[2::], ['窝'])
    #feature['保护_融化加成']=ronghua(linewq)

    PLANTS=['狙', '麦', '坚', '雷', '寒', '嘴', '双', '小', '阳', '喷','魅', '川', '三', '缠', '火', '高', '海', '灯','仙', '叶', '裂', '星', '磁', '卷', '玉', '蒜', '伞', '金','瓜', '机', '曾', '猫', '冰', '吸', '刺', '爆', '飘', '反','若', '奶', '幽', '逆', '藤']
    for plant in PLANTS:
        feature[f'{plant}数']=count_chars(lineup,[plant])
    feature['前窝']=count_chars(lineup[2::],['窝'])
    feature['后胆']=count_chars(lineup[:2:],['胆'])
    '''
    #会产生大量稀疏特征，先不开了
    feature['plant_1']=lineup[0]
    feature['plant_2']=lineup[1]
    feature['plant_3']=lineup[2]
    feature['plant_4']=lineup[3]
    feature['plant_5']=lineup[4]
    '''
    return feature
