import pandas as pd

df = pd.read_csv('train.smi')


df.drop(columns = ['GPX4'], inplace = True)

# df.drop(df.columns[[0]], axis = 1, inplace = True)

print(df)

df.to_csv('train_2.smi', index = False)
