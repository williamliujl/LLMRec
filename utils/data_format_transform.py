# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 01:01
# @Author  : Peilin Zhou
# @FileName: data_format_transform.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
import json
import os
import argparse
import pandas as pd

def filter_and_convert(input_file, target_id):
    filtered_data = []

    output_file_name = os.path.splitext(input_file)[0]  # 移除输入文件的扩展名
    if target_id is not None:
        output_file_name += '_' + str(target_id)  # 将目标ID作为后缀添加到文件名
    else:
        output_file_name += '_all'  # 如果目标ID为None，则将'all'作为后缀添加到文件名

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if target_id is None or target_id=='all' or data['id'] == target_id:
                filtered_data.append({
                    'instruction': data['prompt'],
                    'input': '',
                    'output': data['completion']
                })

    output_file = output_file_name + '.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"格式调整并筛选后的文件已保存为：{output_file}")
def filter_and_convert_rec(input_file, target_id):
    filtered_data = []

    output_file_name = os.path.splitext(input_file)[0]  # 移除输入文件的扩展名
    if target_id is not None:
        output_file_name += '_' + str(target_id)  # 将目标ID作为后缀添加到文件名
    else:
        output_file_name += '_all'  # 如果目标ID为None，则将'all'作为后缀添加到文件名

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if target_id is None or target_id=='all' or data['task_type'] == target_id:
                filtered_data.append({
                    'instruction': data['source'],
                    'input': '',
                    'output': data['target']
                })

    output_file = output_file_name + '.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"格式调整并筛选后的文件已保存为：{output_file}")
def convert_rec(input_file):
    filtered_data = []

    output_file_name = os.path.splitext(input_file)[0]  # 移除输入文件的扩展名


    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if target_id is None or target_id=='all' or data['task_type'] == target_id:
                filtered_data.append({
                    'instruction': data['source'],
                    'input': '',
                    'output': data['target']
                })

    output_file = output_file_name + '.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"格式调整并筛选后的文件已保存为：{output_file}")
def word_count(input_file):
    max_word_count=0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            source_sentence = data['source']
            word_count = len(source_sentence.split())  # 获取句子中按空格划分的词数
            max_word_count = max(max_word_count, word_count)  # 更新最大词数
    print(f"文件中句子的最大词数为: {max_word_count}")

def split_csv_file(file_path, num_parts):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 计算每份数据的大小
    total_rows = len(df)
    rows_per_part = total_rows // num_parts

    # 拆分数据并保存为多个CSV文件
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    file_directory = os.path.dirname(file_path)

    for i in range(num_parts):
        start_index = i * rows_per_part
        end_index = start_index + rows_per_part

        if i == num_parts - 1:
            end_index = total_rows

        part_df = df[start_index:end_index]
        part_file_name = f"{file_name_without_ext}_{i + 1}.csv"
        part_file_path = os.path.join(file_directory, part_file_name)
        part_df.to_csv(part_file_path, index=False)


def merge_csv_files(file_name, output_file):
    # 获取目录下所有以指定文件名开头的CSV文件
    file_directory = os.path.dirname(output_file)
    csv_files = [file for file in os.listdir(file_directory) if file.startswith(file_name) and file.endswith('.csv') and file!=file_name+'.csv']
    csv_files.sort()  # 按文件名排序

    # 合并CSV文件
    merged_df = pd.concat([pd.read_csv(os.path.join(file_directory, file)) for file in csv_files])

    # 保存合并后的CSV文件
    merged_df.to_csv(output_file, index=False)

def compare_csv_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 比较两个DataFrame是否相同
    if df1.equals(df2):
        print("两个CSV文件完全相同")
    else:
        print("两个CSV文件不完全相同")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Filter and convert JSON file.')
    # parser.add_argument('input_file', type=str, help='path to the input JSON file', default='data/train_prompt.json')
    # parser.add_argument('target_id', type=str, nargs='?', default=None, help='target ID for filtering (optional)')
    # args = parser.parse_args()
    #
    # input_file_path = args.input_file
    # target_id = args.target_id
    #
    # filter_and_convert(input_file_path, target_id)
    # file_path = 'data/test.csv'
    # num_parts = 3
    # split_csv_file(file_path, num_parts)
    # merge_csv_files('test', 'data/combine_test.csv')
    # file1='data/test.csv'
    # file2 = 'data/combine_test.csv'
    # compare_csv_files(file1, file2)
    word_count('data/chatglm-data-full/full_train.json')