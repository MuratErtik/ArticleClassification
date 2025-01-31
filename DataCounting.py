import pandas as pd

df = pd.read_csv("example.csv")

datasize = df.shape[0]

print(f"The file has {datasize} data.")

