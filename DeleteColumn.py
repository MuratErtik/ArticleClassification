import pandas as pd

file_path = "dataset.csv"  

df = pd.read_csv(file_path)

filtered_df = df[df["Category"] != "Other"]

output_file = "dataset_filtered.csv"  
filtered_df.to_csv(output_file, index=False)