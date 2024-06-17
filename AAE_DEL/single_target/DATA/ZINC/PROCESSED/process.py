import pandas as pd

df = pd.read_csv('train.smi')

df.drop(['CA9', 'GPX4'], axis = 1, inplace = True)

df.drop(df.columns[[0]], axis = 1, inplace = True)

df.to_csv('train_new.smi')

print(df)
