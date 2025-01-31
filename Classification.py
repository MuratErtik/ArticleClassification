import pandas as pd
import re

# Save the sample CSV file.
input_file = "all6.csv"  
output_file = "dataset.csv"


df = pd.read_csv(input_file)

#keywords with Regex
biology_keywords = r"\b(Bio\w*|Biology\w*|DNA\w*|Genetic\w*|Cancer\w*|Molecular\w*|Cell\w*|Enzyme\w*|Biosynthesis\w*|Microbial\w*|Neuro\w*|Immunology\w*|Plant\w*|Protein\w*|Chemotherap\w*)\b"
chemistry_keywords = r"\b(Chemistry\w*|Chemical\w*|Alcohol\w*|Carbon\w*|Electron\w*|Hydrogen\w*|Magnesium\w*|Nitro\w*|Oxygen\w*|Sulfur\w*)\b"
physics_keywords = r"\b(Electric\w*|Mechanic\w*|Nuclear\w*|Optic\w*|Physic\w*|Quantum\w*|Space\w*|Physical\w*|Newton\w*|Nucleat\w*)\b"
regional_keywords = r"\b(Argentina\w*|Australian\w*|Brazil\w*|Canada\w*|Canadian\w*|Cyprus\w*|Egypt\w*|England\w*|Europe\w*|France\w*|French\w*|Germany\w*|German\w*|India\w*|Iran\w*|Italy\w*|Italian\w*|Norway\w*|Ottoman\w*|Poland\w*|Polish\w*|Russia\w*|Russian\w*|Turkish\w*|Turkey\w*|Ukraine\w*|British\w*)\b"
religion_keywords = r"\b(Catholic\w*|Muslim\w*|Protestant\w*|Religion\w*|Religious\w*|Protests)\b"
technology_keywords = r"\b(Technology\w*|Computer\w*|Internet\w*|Software\w*|Int\w*|Comput\w*)\b"


# Matching between keywords and category
def assign_category(title):
    
    if re.search(biology_keywords, title, re.IGNORECASE):  
        return "Biology"
    elif re.search(chemistry_keywords, title, re.IGNORECASE):  
        return "Chemistry"
    elif re.search(physics_keywords, title, re.IGNORECASE):  
        return "Physics"
    elif re.search(regional_keywords, title, re.IGNORECASE):  
        return "Regional"
    elif re.search(religion_keywords, title, re.IGNORECASE):  
        return "Religion"
    elif re.search(technology_keywords, title, re.IGNORECASE):  
        return "Technolgy"
    
    return "Other"  


df['Category'] = df['Article Name'].apply(assign_category)

df.to_csv(output_file, index=False)

print(f"Completed Successfully.")

