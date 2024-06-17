import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

run_dirs = [

    # Windows
    # f'../RUNS/{directory}' for directory in os.listdir("../RUNS")
    'RUNS/2022-09-07@08-02-22-manshari-desktop-PCBA-FNDS-PSC',
    'RUNS/2022-09-06@10-50-26-manshari-desktop-PCBA-SOPR-PSC',
    # 'RUNS/2022-09-06@16-45-59-manshari-desktop-ZINC-FNDS-PSC',
    # 'RUNS/2022-09-09@06-21-29-manshari-desktop-ZINC-SOPR-PSC',

    # Linux
    # PROJ_DIR / directory for directory in os.listdir(str(PROJ_DIR))
]

df = None
labels = ['FNDS', 'SOPR']
count = 0

sns.set(style='darkgrid')
for idx, run_dir in enumerate(run_dirs):
    finalGenStr = run_dir + '/results/samples_del/new_pop_final.csv'
    finalGenData = pd.read_csv(
        finalGenStr,
        index_col=0,
        skip_blank_lines=True
    )
    # properties = ['qed', 'SAS', 'logP', 'rank']
    properties = ['qed', 'SAS', 'logP']
    # print("properties", properties)

    original_data_length = len(finalGenData)
    print("Number of Rows:", original_data_length)
    finalGenData.dropna(subset=properties, inplace=True)
    finalGenData.reset_index(inplace=True, drop=True)
    post_dropna_data_length = len(finalGenData)
    print("Number of Non-Null Samples:", post_dropna_data_length)
    rank_type = [f"{labels[idx]}"] * post_dropna_data_length
    rt = pd.DataFrame({'rank_type': rank_type})
    # print(rt.head())
    # print(len(rt))
    dataset = pd.concat([rt, finalGenData], axis=1)
    # print(dataset.head())
    # print(len(dataset))
    if df is None:
        df = dataset
    else:
        df = pd.concat([df, dataset]).reset_index(drop=True)


# dataset = pd.read_csv('DATA/PCBA/RANKED/train.csv', index_col=0, skip_blank_lines=True)
# properties = ['qed', "SAS", 'logP']
# dataset.dropna(subset=properties, inplace=True)
# dataset.reset_index(inplace=True, drop=True)
# length = len(dataset)
# # dataset = dataset.loc[:post_dropna_data_length, :]
# r = pd.DataFrame({'rank_type': ['PCBA'] * length})
# dataset = pd.concat([r, dataset], axis=1)
# print(dataset.head())

# df = pd.concat([df, dataset]).reset_index(drop=True)

sns.displot(df, x='qed', hue='rank_type', kind='kde', legend=False)
sns.displot(df, x='SAS', hue='rank_type', kind='kde', legend=False)
sns.displot(df, x='logP', hue='rank_type', kind='kde', legend=False)

plt.legend()
plt.show()
