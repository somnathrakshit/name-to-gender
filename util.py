import pandas as pd

df = pd.read_csv('data/PR.TXT', sep=",", header=None)
df.columns = ["cat", "gender", "year", "name", "count"]

male = df.loc[df['gender'] == 'M']['name']
female = df.loc[df['gender'] == 'F']['name']

male.to_csv('data/names/male.txt', index=False)
female.to_csv('data/names/female.txt', index=False)