import os
import pandas as pd
import warnings
import re

warnings.filterwarnings("ignore")
# rank_reports = ['./reports/' + file for file in os.listdir('./reports/') if file.startswith("RANK_REPORT_45_")]

rank_reports = []
start_year = 1996
end_year = 2017
prefix = "RANK_REPORT_50_"

for year in range(start_year, end_year+1):
    pattern = r"{prefix}{year}_".format(prefix=prefix, year=year)
    regex = re.compile(pattern)
    for file_name in os.listdir('./reports'):
        if regex.match(file_name):
            rank_reports.append('./reports/'+file_name)

print(rank_reports)

full_df = pd.DataFrame()
for report in rank_reports:
    tmp_df = pd.read_csv(report)
    full_df = pd.concat([full_df, tmp_df])

print("full_df")
print(full_df)
print()

rank_cols = full_df.columns
rank_cols = [col_name for col_name in full_df.columns if col_name.endswith("_rank")]
rank_per_cols = pd.DataFrame(rank_cols)

rank_per_cols.set_index(0, inplace=True)
new_index = []
for index in rank_per_cols.index:
    parts = index.split("_")  # "_"를 기준으로 인덱스를 분리하여 리스트로 반환
    new_index.append("_".join(parts[:-2]))  # 뒷 단어 2개를 제외한 나머지 요소를 연결하여 새로운 인덱스 생성
rank_per_cols.index = new_index

for row in rank_per_cols.index.tolist():
    nan_sum = full_df[row].isnull().sum()
    zero_sum = len(full_df.loc[full_df[row] == 0])
    rank_per_cols.loc[row, 'empty'] = zero_sum + nan_sum
    rank_per_cols.loc[row, 'zero'] = zero_sum
    rank_per_cols.loc[row, 'var'] = full_df[row].var()
    rank_per_cols.loc[row, 'avg'] = full_df[row].mean()

rank_per_cols.to_csv('./sample_rank_orig.csv')

# CUT_EMPTY_NUM=54000000
# rank_per_cols = rank_per_cols[rank_per_cols['empty']<CUT_EMPTY_NUM]

# CUT_AVG_NUM=1000000
# rank_per_cols = rank_per_cols[rank_per_cols['avg']<CUT_AVG_NUM]

rank_per_cols = rank_per_cols.sort_values(by='empty')
percentage = 0.7
rows_to_include = int(len(rank_per_cols) * percentage)
rank_per_cols = rank_per_cols.iloc[:rows_to_include]

rank_per_cols = rank_per_cols.sort_values(by='avg')
percentage = 0.2
rows_to_include = int(len(rank_per_cols) * percentage)
rank_per_cols = rank_per_cols.iloc[:rows_to_include]

rank_per_cols['average_rank'] = rank_per_cols['avg'].rank(method='max', ascending=True)
    
print("rank_per_cols")
print(rank_per_cols)
# rank_per_cols.to_csv('sample1.csv')
print() 

max_value = rank_per_cols['average_rank'].max()
min_value = rank_per_cols['average_rank'].min()
rank_per_cols['norm_rank'] = (rank_per_cols['average_rank'] - min_value) / (max_value - min_value) * 10
 
print("rank_per_cols")
print(rank_per_cols)
# rank_per_cols.to_csv('sample2.csv')
print() 
 
max_value = rank_per_cols['var'].max()
min_value = rank_per_cols['var'].min()
rank_per_cols['norm_var'] = (rank_per_cols['var'] - min_value) / (max_value - min_value) * 10

rank_per_cols['weight_base'] = (10-rank_per_cols['norm_var']) + (10-rank_per_cols['norm_rank'])

print("rank_per_cols")
print(rank_per_cols)
print() 

max_value = rank_per_cols['weight_base'].max()
min_value = rank_per_cols['weight_base'].min()
rank_per_cols['weight'] = (rank_per_cols['weight_base'] - min_value) / (max_value - min_value) * 10
    
rank_per_cols.to_csv('./sample_rank.csv')
print(rank_per_cols)

plan_sample = pd.DataFrame()
# rank_per_cols.set_index(0, inplace=True)
# plan_sample['key'] = rank_per_cols[rank_per_cols.columns[0]]
plan_sample['weight'] = rank_per_cols['weight']
plan_sample['key_dir'] = 'low'
plan_sample['diff'] = '10'
plan_sample['base'] = '>'
plan_sample['base_dir'] = '0'
plan_sample.to_csv('./sample_plan.csv', index_label=['key'])


