import pandas as pd

df = pd.read_csv('train.smi')
df = df.smiles.tolist()


with open(r'train_jtvae.txt', 'a') as fp:
    for s in df:
        fp.write(s + '\n')
