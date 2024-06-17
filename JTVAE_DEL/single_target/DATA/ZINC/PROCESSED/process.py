import pandas as pd

df = pd.read_csv('train_three.smi')
df.drop(['CA9'], axis = 1, inplace = True)
df.drop(df.columns[[0]], axis = 1, inplace = True)
df.to_csv('train_dual.smi')
print(df)

