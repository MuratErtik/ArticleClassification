import re
import pandas as pd


file_path = "dataset_filtered2.csv"  
df = pd.read_csv(file_path)

def clean_text(text):
    
    text = re.sub(r"<.*?>", "", text)
    
    text = re.sub(r"[^\w\s]", "", text)  
    
    
    text = re.sub(r"[\n\t\r]", " ", text)
    
    
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

df["Content"] = df["Content"].apply(clean_text)

df["Article name"] = df["Article name"].apply(clean_text)

output_file = "dataset_filtered2.csv"
df.to_csv(output_file, index=False)
