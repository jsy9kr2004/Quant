import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
rank_reports = ['./reports/' + file for file in os.listdir('./reports/') if file.startswith("RANK_REPORT_46_")]

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

for row in rank_per_cols.index.tolist():
    nan_sum = full_df[row].isnull().sum()
    zero_sum = len(full_df.loc[full_df[row] == 0])
    rank_per_cols.loc[row, 'empty'] = zero_sum + nan_sum
    rank_per_cols.loc[row, 'zero'] = zero_sum
    rank_per_cols.loc[row, 'var'] = full_df[row].var()
    rank_per_cols.loc[row, 'avg'] = full_df[row].mean()

CUT_EMPTY_NUM=3600
rank_per_cols = rank_per_cols[rank_per_cols['empty']<CUT_EMPTY_NUM]

CUT_AVG_NUM=924
rank_per_cols = rank_per_cols[rank_per_cols['avg']<CUT_AVG_NUM]

rank_per_cols['average_rank'] = rank_per_cols['avg'].rank(method='max', ascending=True)
    
print("rank_per_cols")
print(rank_per_cols)
rank_per_cols.to_csv('sample1.csv')
print() 

max_value = rank_per_cols['average_rank'].max()
min_value = rank_per_cols['average_rank'].min()
rank_per_cols['norm_rank'] = (rank_per_cols['average_rank'] - min_value) / (max_value - min_value) * 10
 
print("rank_per_cols")
print(rank_per_cols)
rank_per_cols.to_csv('sample2.csv')
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



