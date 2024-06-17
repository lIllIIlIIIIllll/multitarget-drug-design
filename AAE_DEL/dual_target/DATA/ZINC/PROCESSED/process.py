import pandas as pd

df = pd.read_csv('train.smi')
df.drop(['GPX4'], axis = 1, inplace = True)
df.drop(df.columns[[0]], axis = 1, inplace = True)
df.to_csv('train_lpa1.smi')
print(df)

