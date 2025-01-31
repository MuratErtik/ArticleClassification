# ArticleClassification Repository

## 1. Project Overview
This project focuses on text classification and text summarization using Transformer-based models (BERT, ALBERT, DistilBERT, RoBERTa, GPT-2). It includes data preprocessing, model training, evaluation, and performance analysis.

## 2. Repository Contents
📂 **data/** – Raw and processed datasets used for training and evaluation.  
📂 **notebooks/** – Jupyter/Colab notebooks for data preprocessing, model training, and evaluation.  
📂 **models/** – Saved models and checkpoints.  
📂 **results/** – Performance metrics, confusion matrices, and ROC curves.  
📂 **scripts/** – Python scripts for data preprocessing, training, and inference.  
📄 **README.md** – Project details, requirements, and usage instructions.  
📄 **report.pdf** – Final report including data processing, models, and results.  

## 3. Key Features
✅ Minimum 5,000 records per class for classification tasks.  
✅ Data preprocessing: Tokenization, stopword removal, lemmatization, stemming, and additional techniques.  
✅ Five Transformer models trained and compared.  
✅ Performance evaluation: Accuracy, Precision, Recall, Sensitivity, Specificity, F1-Score, AUC.  
✅ Confusion matrices and ROC curves for model analysis.  
✅ Training & Inference time measurement for all models.  

This repository provides all necessary resources for conducting NLP classification and summarization with deep learning models. 🚀


## 4. Web Scraping

### 1. Extracting Page Links

Article links from Wikipedia's "All Pages" section are analyzed using BeautifulSoup. The URLs are retrieved from the "mw-allpages-body" HTML div element.

### 2. Extracting Article Content

Each collected link is visited, and the article content is extracted:

Article Title: The page title is stored.

Article Content: Important sections such as headings and paragraphs are filtered using the h2, h3, p HTML tags.

Error Handling: If the content cannot be retrieved or the page does not exist, error messages are displayed.

### 3. Saving Data to CSV

All extracted articles are saved in CSV format. Each row contains:

"Article Name" – The title of the article.

"Content" – The extracted text from the article.

This process enables automated data collection (web scraping) and dataset creation, which is essential for training machine learning models with large-scale text data.




