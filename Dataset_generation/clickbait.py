# %%
# Clickbait Detection with Method Attribution - Full Implementation
# Configuration
# %pip install openai transformers datasets scikit-learn pandas tensorflow dotenv seaborn matplotlib tf-keras ipywidgets requests
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, multilabel_confusion_matrix
from transformers import BertTokenizer, TFBertForSequenceClassification
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv

load_dotenv()

client = requests.session()
client.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}


def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found,
    or there are less than 2 characters, return the string unchanged.
    """
    if (len(s) >= 2 and s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s


def openai_request(prompt):
    choices = None
    while not choices:
        response = client.post(url=f'{os.getenv("OPENAI_API_URL")}/chat/completions',
                               json=
                               {
                                   "model": os.getenv('OPENAI_API_MODEL'),
                                   "messages": [{
                                       "role": "user",
                                       "content": prompt
                                   }]
                               })
        choices = response.json().get('choices', None)
        if not choices:
            continue

    return dequote(choices[0]["message"]["content"].strip())


# 1. Dataset Generation
## 1.1 Create clickbait methods catalog
METHODS_CATALOG = [
    "Curiosity Gap", "Exaggeration", "Emotional Triggers",
    "Sensationalism", "Lists/Superlatives", "Ambiguous References",
    "Direct Appeals", "Unfinished Narratives", "Unexpected Associations",
    "Provocative Questions"
]
if not Path('clickbait_methods.json').exists():
    with open('clickbait_methods.json', 'w') as f:
        json.dump(METHODS_CATALOG, f)

## 1.2 Load real news dataset
real_news_df = pd.read_csv('news_data.csv')
real_news_df['title'] = real_news_df['title'].apply(lambda cell: cell.encode('ASCII', 'ignore').decode('ASCII'))
real_news = list(real_news_df['title'])


# 1.3 Modified synthetic data generation with batching
def generate_clickbait_batch(batch):
    prompt = f"""
    For each headline below, create a clickbait version using ONLY the specified methods.
    Do not alter factual content, only style.
    Return JSON list without any code blocks or wrappings with items containing:
    - "original": original headline,
    - "methods_used": methods applied,
    - "clickbait": generated text

    Headlines with methods: 
    {json.dumps([{'original': item['original'], 'methods': item['methods']} for item in batch])}
    """

    attempts = 0
    while attempts < 3:
        try:
            response = openai_request(prompt)
            return json.loads(response)
        except json.JSONDecodeError:
            attempts += 1
    raise Exception("Failed to parse valid JSON after 3 attempts")


# Generate dataset with batching
batch_size = 10  # Adjust based on model context window
dataset = []

if Path('clickbait_dataset.csv').exists():
    df = pd.read_csv('clickbait_dataset.csv')
else:
    for i in tqdm(range(0, len(real_news), batch_size), desc="Generating Clickbait"):
        print(f"{i}/{len(real_news)}")
        batch = real_news[i:i + batch_size]
        batch_data = []

        for news in batch:
            k = np.random.randint(1, 6)
            selected_methods = np.random.choice(METHODS_CATALOG, k, replace=False).tolist()
            batch_data.append({
                'original': news,
                'methods': selected_methods
            })

        results = generate_clickbait_batch(batch_data)

        for res in results:
            method_vector = [1 if m in res['methods_used'] else 0 for m in METHODS_CATALOG]
            dataset.append({
                'source': res['original'],
                'clickbait': res['clickbait'],
                'methods': method_vector
            })

    df = pd.DataFrame(dataset)
    df.to_csv('clickbait_dataset.csv', index=False)
# %%
# 2. Evaluation Framework
## 2.1 Prepare data splits
X_detection = df[['source', 'clickbait']].values.flatten()
y_detection = [0] * len(df) + [1] * len(df)

X_attribution = df['clickbait'].tolist()
y_attribution = np.array(df['methods'].tolist())

# Train/Test split
X_det_train, X_det_test, y_det_train, y_det_test = train_test_split(
    X_detection, y_detection, test_size=0.2, random_state=42, stratify=y_detection)

X_att_train, X_att_test, y_att_train, y_att_test = train_test_split(
    X_attribution, y_attribution, test_size=0.2, random_state=42)
# %%
# 2.2 Detection Models
## 2.2.1 Logistic Regression Baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_det_train_tfidf = tfidf.fit_transform(X_det_train)
X_det_test_tfidf = tfidf.transform(X_det_test)

# Train
lr = LogisticRegression(max_iter=1000)
lr.fit(X_det_train_tfidf, y_det_train)

# Evaluate
lr_preds = lr.predict(X_det_test_tfidf)
print("Logistic Regression Detection Results:")
print(f"Accuracy: {accuracy_score(y_det_test, lr_preds):.2f}")
print(f"F1 Score: {f1_score(y_det_test, lr_preds):.2f}")

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_det_test, lr_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()
# %%
## 2.2.2 BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Tokenize
X_det_train_bert = tokenizer(X_det_train.tolist(), padding=True, truncation=True, return_tensors='tf')
X_det_test_bert = tokenizer(X_det_test.tolist(), padding=True, truncation=True, return_tensors='tf')

# Train
bert_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bert_model.fit(X_det_train_bert.data, np.array(y_det_train), epochs=3, batch_size=16)

# Evaluate
# Predict and process correctly
logits = bert_model.predict(X_det_test_bert.data).logits
probabilities = tf.sigmoid(logits).numpy().flatten()  # Apply sigmoid
bert_preds = (probabilities > 0.5).astype(int)  # Threshold at 0.5
print("BERT Detection Results:")
print(f"Accuracy: {accuracy_score(y_det_test, bert_preds):.2f}")
print(f"F1 Score: {f1_score(y_det_test, bert_preds):.2f}")

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_det_test, bert_preds), annot=True, fmt='d', cmap='Blues')
plt.title('BERT Confusion Matrix')
plt.show()
# %%
# 2.3 Attribution Models
## 2.3.1 Multi-label BERT
att_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
att_model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(METHODS_CATALOG),
    problem_type="multi_label_classification"
)

# Tokenize
X_att_train_tok = att_tokenizer(X_att_train, padding=True, truncation=True, return_tensors='tf')
X_att_test_tok = att_tokenizer(X_att_test, padding=True, truncation=True, return_tensors='tf')

# Train
att_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
att_model.fit(X_att_train_tok.data, y_att_train, epochs=3, batch_size=16)

# Evaluate
att_preds = (att_model.predict(X_att_test_tok.data).logits > 0.5).astype(int)
print("BERT Attribution Results:")
print(f"Micro F1: {f1_score(y_att_test, att_preds, average='micro'):.2f}")
print(f"Macro F1: {f1_score(y_att_test, att_preds, average='macro'):.2f}")

# Confusion Matrix (Example for first method)
plt.figure(figsize=(12, 6))
for i, method in enumerate(METHODS_CATALOG[:2]):  # Show first 2 methods
    cm = multilabel_confusion_matrix(y_att_test, att_preds)[i]
    plt.subplot(1, 2, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{method} Confusion Matrix')
plt.tight_layout()
plt.show()


# %%
## 2.3.2 Openai_Model Zero-Shot Attribution
def openai_model_attribution(text):
    prompt = f"""
    Identify which of these methods were used in the headline:
    {', '.join(METHODS_CATALOG)}

    Headline: {text}
    """
    content = openai_request(prompt)
    return [1 if m in content else 0 for m in METHODS_CATALOG]


# Create a mapping from clickbait text to its index in X_att_test
clickbait_to_index = {clickbait: idx for idx, clickbait in enumerate(X_att_test)}

# Get sampled data
sample_att = df.loc[df['clickbait'].isin(X_att_test)].sample(10, random_state=42)

# Get true labels using clickbait text alignment
sample_indices = [clickbait_to_index[cb] for cb in sample_att['clickbait']]
y_true = y_att_test[sample_indices]

# Generate Openai_Model predictions
sample_att['openai_model_methods'] = sample_att['clickbait'].apply(openai_model_attribution)
y_openai_model = np.array(sample_att['openai_model_methods'].tolist())

# Plot confusion matrices
plt.figure(figsize=(12, 6))
for i, method in enumerate(METHODS_CATALOG[:2]):
    cm = confusion_matrix(y_true[:, i], y_openai_model[:, i])
    plt.subplot(1, 2, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Openai_Model {method} Confusion Matrix')
plt.tight_layout()
plt.show()
# %%
# 3. Results Compilation
results = {
    "Detection Results": [
        {
            "Model": "Logistic Regression",
            "Accuracy": accuracy_score(y_det_test, lr_preds),
            "F1": f1_score(y_det_test, lr_preds)
        },
        {
            "Model": "BERT",
            "Accuracy": accuracy_score(y_det_test, bert_preds),
            "F1": f1_score(y_det_test, bert_preds)
        }
    ],
    "Attribution Results": [
        {
            "Model": "Multi-label BERT",
            "Micro F1": f1_score(y_att_test, att_preds, average='micro'),
            "Macro F1": f1_score(y_att_test, att_preds, average='macro')
        }
    ]
}

pd.DataFrame(results['Detection Results']).to_csv('detection_results.csv')
pd.DataFrame(results['Attribution Results']).to_csv('attribution_results.csv')
