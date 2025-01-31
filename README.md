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



## GPT Model Training and Evaluation

This section demonstrates the training and evaluation results of the GPT model on the dataset. The dataset was split into 80% training and 20% validation data in a random manner. Below are the training and evaluation metrics for the model.

### Training Process:

The model was trained over 5 epochs, with the following results for each epoch:

#### Epoch 1/5:
- **Training Loss**: 0.5559
- **Validation Loss**: 0.2847
- **Training Time**: 1059.12 seconds

#### Epoch 2/5:
- **Training Loss**: 0.2647
- **Validation Loss**: 0.2286
- **Training Time**: 937.92 seconds

#### Epoch 3/5:
- **Training Loss**: 0.1904
- **Validation Loss**: 0.2271
- **Training Time**: 958.37 seconds

#### Epoch 4/5:
- **Training Loss**: 0.1437
- **Validation Loss**: 0.2193
- **Training Time**: 949.22 seconds

#### Epoch 5/5:
- **Training Loss**: 0.1115
- **Validation Loss**: 0.2230
- **Training Time**: 951.38 seconds

#### Total Training Time:
- **4856.01 seconds**

### Evaluation Metrics:

After training the model, the following evaluation metrics were obtained:

- **Accuracy**: 0.9371
- **Precision**: 0.9371
- **Recall**: 0.9371
- **F1 Score**: 0.9370
- **Sensitivity**: 0.9802
- **Specificity**: 0.9655
- **AUC (Area Under the Curve)**: 0.9944

### Inference Time:

- **Inference Time**: 0.0389 seconds

### Summary:

The model was trained effectively with reasonable loss reduction over 5 epochs. The evaluation metrics show a well-performing model, with an accuracy of 93.71% and high values for Precision, Recall, F1 Score, Sensitivity, Specificity, and AUC. The inference time is also quite low, making the model suitable for real-time applications.

