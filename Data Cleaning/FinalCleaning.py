import pandas as pd
import re


input_file = "all3.csv"  
output_file = "all4.csv"  


df = pd.read_csv(input_file)

def clean_empty_sections(content):
    if pd.isna(content):  
        return content
    
   
    
    pattern = r"(== .*? ==)(?:\n\s*\n|\n?$)"  
    
    
    cleaned_content = re.sub(pattern, "", content, flags=re.MULTILINE)
    return cleaned_content.strip()


def filter_short_content(content):
    if pd.isna(content):  
        return None
    word_count = len(content.split())  
    return content if word_count >= 20 else None 


df['Content'] = df['Content'].apply(clean_empty_sections)
df['Content'] = df['Content'].apply(filter_short_content)

df = df.dropna(subset=['Content'])

df.to_csv(output_file, index=False)

