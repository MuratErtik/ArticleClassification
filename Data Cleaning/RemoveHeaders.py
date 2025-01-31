import pandas as pd
import re

input_file = "databackup/all.csv"  
output_file = "all2.csv"  


df = pd.read_csv(input_file)

remove_headers = [
    "== See also ==",
    "== Notes ==",
    "== References ==",
    "== Sources ==",
    "== External links =="
]

def remove_unwanted_sections(content):
    if pd.isna(content):  
        return content

 
    for header in remove_headers:
        
        content = re.sub(rf"{re.escape(header)}.*?(?=(==|$))", "", content, flags=re.DOTALL)

    
    content = content.strip()
    return content


df['Content'] = df['Content'].apply(remove_unwanted_sections)


df.to_csv(output_file, index=False)