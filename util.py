import pandas as pd

df = pd.read_csv('data/yob2016.txt', sep=",", header=None)
df.columns = ["name", "gender", "count"]

male = df.loc[df['gender'] == 'M']['name']
female = df.loc[df['gender'] == 'F']['name']

male.to_csv('data/names/male.txt', index=False)
female.to_csv('data/names/female.txt', index=False)