import os
import pandas as pd


folder_path = "/Users/muratertik/xxxxxx/data/xxxxxxxxxx"  

csv_files = []

for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        if file_name.endswith(".csv"):
            file_path = os.path.join(root, file_name)
            csv_files.append(pd.read_csv(file_path))

merged_csv = pd.concat(csv_files, ignore_index=True)

output_file = os.path.join(folder_path, "all.csv")

merged_csv.to_csv(output_file, index=False)
