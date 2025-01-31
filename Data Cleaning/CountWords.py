import pandas as pd


input_file = "dataset_filtered.csv"  
output_file = "dataset_filtered2.csv"  


df = pd.read_csv(input_file)

def has_minimum_words(content):
    if pd.isna(content): 
        return False
    
    word_count = len(content.split())
    return word_count >= 100  #


df = df[df['Content'].apply(has_minimum_words)]


df.to_csv(output_file, index=False)

