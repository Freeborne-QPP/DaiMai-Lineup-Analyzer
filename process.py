import re
import csv
import json
import glob
from pathlib import Path
import os

def try_decode_file(file_path):
    """尝试用不同编码读取文件"""
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"  ✓ 成功使用 {encoding} 编码读取: {file_path}")
            return content, encoding
        except UnicodeDecodeError:
            continue
    
    # 如果所有编码都失败，尝试二进制读取
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        # 尝试猜测编码
        import chardet
        result = chardet.detect(content)
        detected_encoding = result['encoding']
        print(f"  ✓ 检测到编码: {detected_encoding} (置信度: {result['confidence']:.2f}): {file_path}")
        return content.decode(detected_encoding, errors='ignore'), detected_encoding
    except Exception as e:
        print(f"  ✗ 无法解码文件: {file_path}")
        return None, None

def process_plant_string(plant_str):
    """处理单个植物字符串，按照规则①②③处理"""
    if not plant_str or plant_str == "":
        return ""
    
    # 规则①：如果同时有"雷"和其他字，优先保留其他字
    if "雷" in plant_str and len(plant_str) > 1:
        # 移除"雷"字
        result = plant_str.replace("雷", "")
        # 如果移除后还有字，继续处理
        if result:
            plant_str = result
    
    # 规则②：如果同时有"窝"和其他字，优先保留其他字
    if "窝" in plant_str and len(plant_str) > 1:
        # 移除"窝"字
        result = plant_str.replace("窝", "")
        # 如果移除后还有字，继续处理
        if result:
            plant_str = result
    
    # 规则③：有多个字时，取第一个字
    if len(plant_str) >= 1:
        return plant_str[0]
    else:
        return ""

def extract_array_from_string(line_str):
    """从字符串中提取数组部分"""
    # 找到第一个'[['和最后一个']]'
    start = line_str.find('[[')
    end = line_str.rfind(']]')
    
    if start != -1 and end != -1 and end > start:
        array_str = line_str[start:end+2]
        return array_str
    return ""

def parse_line(line, file_name=""):
    """解析单行数据"""
    try:
        # 按制表符分割
        parts = line.strip().split('\t')
        
        if len(parts) < 7:  # 至少应该有时间戳+5个成绩+植物数组
            return None
        
        timestamp = parts[0]
        scores = []
        
        # 解析5个成绩
        for i in range(1, 6):
            try:
                score_str = parts[i].strip()
                if score_str:
                    score = int(score_str)
                    scores.append(score)
                else:
                    scores.append(0)
            except (ValueError, IndexError):
                scores.append(0)
        
        # 检查是否全为0（规则⑥）
        if all(score == 0 for score in scores):
            return None
        
        # 提取数组字符串（合并剩余部分）
        array_str = extract_array_from_string(line)
        
        if not array_str:
            # 如果没有找到标准数组格式，尝试从第6部分开始合并
            array_str = '\t'.join(parts[6:])
        
        # 尝试解析数组
        plant_arrays = []
        
        # 方法1: 尝试JSON解析
        try:
            json_str = array_str.replace("'", '"')
            plant_arrays = json.loads(json_str)
        except:
            # 方法2: 使用正则表达式手动解析
            try:
                # 匹配每个阵容数组
                lineup_pattern = r'\[(.*?)\]'
                all_matches = re.findall(lineup_pattern, array_str)
                
                # 提取5个阵容
                for match in all_matches[:5]:
                    if match.strip():
                        # 匹配每个植物字符串
                        plant_pattern = r"'([^']*)'"
                        plants = re.findall(plant_pattern, match)
                        plant_arrays.append(plants)
                    else:
                        plant_arrays.append([])
                
                # 如果解析出的阵容不足5个，补充空数组
                while len(plant_arrays) < 5:
                    plant_arrays.append([])
                    
            except Exception:
                return None
        
        return {
            'timestamp': timestamp,
            'scores': scores,
            'plants': plant_arrays,
            'source_file': file_name
        }
        
    except Exception:
        return None

def process_single_file(file_path):
    """处理单个文件，返回处理后的数据列表"""
    processed_data = []
    file_name = os.path.basename(file_path)
    
    # 读取文件内容
    try:
        content, used_encoding = try_decode_file(file_path)
        if content is None:
            print(f"  ✗ 跳过文件: {file_path} (无法解码)")
            return processed_data
            
        lines = content.splitlines()
        print(f"  共读取 {len(lines)} 行数据")
    except Exception as e:
        print(f"  ✗ 读取文件失败: {file_path} - {e}")
        return processed_data
    
    valid_records = 0
    skipped_records = 0
    
    for line_num, line in enumerate(lines, 1):
        # 跳过空行
        if not line.strip():
            continue
        
        parsed_data = parse_line(line, file_name)
        if parsed_data is None:
            skipped_records += 1
            continue
        
        timestamp = parsed_data['timestamp']
        scores = parsed_data['scores']
        plants = parsed_data['plants']
        source_file = parsed_data['source_file']
        
        # 处理每个阵容（1-5路）
        for i in range(5):
            score = scores[i]
            lineup_idx = i  # 对应植物数组中的索引
            
            if lineup_idx < len(plants):
                raw_lineup = plants[lineup_idx]
            else:
                raw_lineup = []
            
            # 只取前五个植物（规则④）
            lineup_to_process = raw_lineup[:5]
            
            # 处理每个植物字符串
            processed_plants = []
            for plant_str in lineup_to_process:
                if plant_str and plant_str.strip():  # 非空字符串
                    processed = process_plant_string(plant_str.strip())
                    if processed:
                        processed_plants.append(processed)
                    else:
                        processed_plants.append("空")  # 处理结果为空的占位符
                else:
                    processed_plants.append("空")  # 空字符串的占位符
            
            # 如果植物数量不足5个，用"空"填充
            while len(processed_plants) < 5:
                processed_plants.append("空")
            
            # 创建一行数据
            row = [
                timestamp,            # 时间戳
                str(i + 1),           # 路数（规则⑤）
                str(score),           # 该路成绩
                ' '.join(processed_plants),  # 处理后的植物序列，用空格分隔
                source_file           # 来源文件名
            ]
            
            # 添加原始植物信息（用于调试）
            raw_plants_str = ' '.join([str(p) if p and p.strip() else "空" for p in raw_lineup[:5]])
            row.append(raw_plants_str)
            
            processed_data.append(row)
            valid_records += 1
    
    print(f"  有效记录: {valid_records}, 跳过记录: {skipped_records}")
    return processed_data

def find_record_files():
    """查找当前目录下所有record开头的txt文件"""
    # 使用glob模块查找文件
    record_files = glob.glob("record*.txt")
    
    # 同时查找子目录中的record文件
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.startswith("record") and file.endswith(".txt"):
                full_path = os.path.join(root, file)
                if full_path not in record_files:
                    record_files.append(full_path)
    
    # 去除可能的重复
    record_files = list(set(record_files))
    
    return record_files

def main():
    """主函数"""
    print("=" * 60)
    print("植物大战僵尸数据转换工具 - 批量处理模式")
    print("=" * 60)
    
    # 查找所有record文件
    print("\n正在查找record开头的txt文件...")
    record_files = find_record_files()
    
    if not record_files:
        print("✗ 未找到任何以'record'开头的txt文件！")
        print("请确保文件在当前目录或子目录中。")
        return
    
    print(f"✓ 找到 {len(record_files)} 个文件:")
    for i, file in enumerate(record_files, 1):
        print(f"  {i}. {file}")
    
    # 询问是否处理所有文件
    response = input("\n是否处理所有文件？(y/n，默认y): ").strip().lower()
    if response == 'n':
        # 让用户选择要处理的文件
        print("\n请输入要处理的文件编号（用逗号分隔，如：1,3,5）:")
        choices = input("选择: ").strip()
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in choices.split(',')]
            selected_files = [record_files[i] for i in selected_indices if 0 <= i < len(record_files)]
            if not selected_files:
                print("✗ 选择无效，将处理所有文件")
                selected_files = record_files
            else:
                record_files = selected_files
                print(f"✓ 将处理 {len(record_files)} 个文件")
        except:
            print("✗ 输入格式错误，将处理所有文件")
    
    # 处理所有文件
    all_processed_data = []
    total_records = 0
    
    print(f"\n开始处理 {len(record_files)} 个文件...")
    print("-" * 40)
    
    for i, file_path in enumerate(record_files, 1):
        print(f"处理文件 {i}/{len(record_files)}: {file_path}")
        file_data = process_single_file(file_path)
        all_processed_data.extend(file_data)
        total_records += len(file_data)
        print(f"  当前总记录数: {total_records}")
        print("-" * 40)
    
    # 写入CSV文件
    output_file = "record.csv"
    
    # 询问是否覆盖已存在文件
    if Path(output_file).exists():
        response = input(f"\n文件 '{output_file}' 已存在，是否覆盖？(y/n): ").strip().lower()
        if response != 'y':
            new_name = input("请输入新的输出文件名: ").strip()
            if new_name:
                output_file = new_name
            else:
                output_file = "record_processed.csv"
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['时间戳', '路数', '成绩', '处理后植物序列', '来源文件', '原始植物序列'])
            
            # 写入数据
            writer.writerows(all_processed_data)
        
        print(f"\n✓ 处理完成！")
        print(f"  总文件数: {len(record_files)}")
        print(f"  总记录数: {total_records}")
        print(f"  结果已保存到: {output_file}")
        
        # 显示前几行数据作为示例
        if all_processed_data:
            print("\n前5行数据示例:")
            for i in range(min(5, len(all_processed_data))):
                print(f"  行{i+1}: {all_processed_data[i]}")
        else:
            print("警告: 没有处理出任何数据！")
            
    except Exception as e:
        print(f"✗ 写入CSV文件时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 提供额外选项
    print("\n" + "=" * 60)
    print("可选操作:")
    print("1. 查看输出文件的前10行")
    print("2. 退出程序")
    
    choice = input("请选择 (1 或 2): ").strip()
    if choice == '1' and Path(output_file).exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"\n{output_file} 前10行内容:")
                for i, line in enumerate(lines[:11]):  # 包括表头
                    print(f"行{i}: {line.strip()}")
        except Exception as e:
            print(f"读取输出文件时出错: {e}")

if __name__ == "__main__":
    main()
