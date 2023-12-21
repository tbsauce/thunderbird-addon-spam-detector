import os
import math
import nltk
import tqdm
import joblib
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models.fasttext import FastText
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import datetime
from dateutil import parser
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def categorize_date(date_str):
    try:
        date_obj = parser.parse(date_str)
        day_start = datetime.time(9, 0, 0)
        day_end = datetime.time(18, 0, 0)

        if date_obj.weekday() < 5:  # Weekday
            return 0 if day_start <= date_obj.time() <= day_end else 1
        else:  # Weekend
            return 2 if day_start <= date_obj.time() <= day_end else 3
    except Exception as e:
        print(f"Error parsing date: {date_str} - {e}")
        return None

def extract_domain(email):
    if isinstance(email, str) and '@' in email:
        return email.split('@')[1].split('>')[0]
    return ''

def preprocess_text(text, lemmatizer, stop_words):
    # Handle cases where text is not a string
    if not isinstance(text, str):
        return ""

    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token).lower() for token in tokens if token.isalpha() and token.lower() not in stop_words and len(lemmatizer.lemmatize(token)) > 2])

# Function to convert text to FastText features
def text_to_fasttext_features(text, model):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Read the CSV file
df = pd.read_csv('training_data.csv')

# Preprocess the data
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('portuguese'))
df['Content'] = df['Content'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
df['Subject'] = df['Subject'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
df['Combined_Text'] = df['Subject'] + ' ' + df['Content']

df['Sender'] = df['Sender'].apply(extract_domain)
df['Reply-To'] = df['Reply-To'].apply(extract_domain)
df['Return-Path'] = df['Return-Path'].apply(extract_domain)
df['Date'] = df['Date'].apply(categorize_date)

# Encoding categorical features
df[['Sender', 'Reply-To', 'Return-Path']] = df[['Sender', 'Reply-To', 'Return-Path']].astype('category')
df[['Sender', 'Reply-To', 'Return-Path']] = df[['Sender', 'Reply-To', 'Return-Path']].apply(lambda x: x.cat.codes)

enc = OneHotEncoder(max_categories=5)
enc_data = pd.DataFrame(enc.fit_transform(df[['Sender', 'Reply-To', 'Return-Path']]).toarray()) 
df = df.join(enc_data)

# Train FastText Model
fasttext_model = FastText(vector_size=10, window=3, min_count=1)
fasttext_model.build_vocab(corpus_iterable=df['Combined_Text'].apply(lambda x: x.split()))
fasttext_model.train(corpus_iterable=df['Combined_Text'].apply(lambda x: x.split()), total_examples=len(df), epochs=10)

# Apply the function to your dataframe
df['FastText_Features'] = df['Combined_Text'].apply(lambda x: text_to_fasttext_features(x, fasttext_model))
fasttext_features = np.stack(df['FastText_Features'].values)

# Train-Test Split
y = df['Label']
X = df.drop(['Label', 'Content', 'Subject', 'Combined_Text','FastText_Features'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(np.hstack((X, fasttext_features)), y, stratify=y, test_size=0.2, random_state=42)

# Define and train classifiers
clfs = [
    ('LR', LogisticRegression(random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('NB', GaussianNB()),
    ('MLP', MLPClassifier(random_state=42)),
    ('RFC', RandomForestClassifier(random_state=42))
]

for label, clf in clfs:
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    mcc = matthews_corrcoef(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    print(f'{label:3} Accuracy: {acc:.2f}, F1 Score: {f1:.2f}, MCC: {mcc:.2f}')    


# Save the model and encoder
joblib.dump(clfs[-1][1], "../model.pkl")
joblib.dump(enc, "../hot_encoder.pkl")
joblib.dump(fasttext_model, "../fasttext.pkl")
