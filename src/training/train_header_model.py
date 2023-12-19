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
plt.rcParams['figure.figsize'] = [15, 15]
plt.rcParams['figure.dpi'] = 72
import seaborn as sns
import seaborn.objects as so
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

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

from sklearn.preprocessing import OneHotEncoder
import datetime
from dateutil import parser


def categorize_date(date_str):
    try:
        # Convert string to datetime object using dateutil.parser
        date_obj = parser.parse(date_str)

        # Define day hours (6 AM to 6 PM)
        day_start = datetime.time(9, 0, 0)
        day_end = datetime.time(18, 0, 0)

        # Check if it's a weekday or weekend
        if date_obj.weekday() < 5:  # Weekday
            if day_start <= date_obj.time() <= day_end:
                return 0  # Weekday (Day)
            else:
                return 1  # Weekday (Night)
        else:  # Weekend
            if day_start <= date_obj.time() <= day_end:
                return 2  # Weekend (Day)
            else:
                return 3  # Weekend (Night)
    except Exception as e:
        print(f"Error parsing date: {date_str} - {e}")
        return None  # or a default category
    
def extract_domain(email):
    if isinstance(email, str) and '@' in email:
        return email.split('@')[1].split('>')[0]
    return ''


# Read the CSV file
df = pd.read_csv('/home/sauce/thunderbird-addon-spam-detector/src/datasets/myDataset.csv')

# Apply the function to extract domains
df['Return-Path_Domain'] = df['Return-Path'].apply(extract_domain)
df['Sender_Domain'] = df['Sender'].apply(extract_domain)

# Compare and assign values
df['Similar-Return'] = df.apply(lambda row: 1 if row['Return-Path_Domain'] == row['Sender_Domain'] else 0, axis=1)

# Optionally, you can remove the temporary columns if they are not needed
df.drop(['Return-Path_Domain', 'Sender_Domain'], axis=1, inplace=True)

# Extracting domains from email addresses
df['Sender'] = df['Sender'].apply(lambda x: '@' + x.split('@')[1][:-1] if isinstance(x, str) and '@' in x else '')
df['Reply-To'] = df['Reply-To'].apply(lambda x: '@' + x.split('@')[1][:-1] if isinstance(x, str) and '@' in x else '')
df['Return-Path'] = df['Return-Path'].apply(lambda x: '@' + x.split('@')[1][:-1] if isinstance(x, str) and '@' in x else '')

df['Date'] = df['Date'].apply(categorize_date)

# print(df.head())

y = df['Label'].tolist()
df['Sender'] = df['Sender'].astype('category')
df['Reply-To'] = df['Reply-To'].astype('category')
df['Return-Path'] = df['Return-Path'].astype('category')

df['Sender'] = df['Sender'].cat.codes
df['Reply-To'] = df['Reply-To'].cat.codes
df['Return-Path'] = df['Return-Path'].cat.codes

enc = OneHotEncoder(max_categories=5)

enc_data = pd.DataFrame(enc.fit_transform( df[['Sender', 'Reply-To', 'Return-Path']]).toarray()) 

df = df.join(enc_data)
df.columns = df.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(df, y, stratify=y, test_size=0.2, random_state=42)

print(f'Training Data : {len(X_train)}')
print(f'Testing Data  : {len(X_test)}')

# define the list of classifiers
clfs = [
    ('LR', LogisticRegression(random_state=42, multi_class='auto', max_iter=1000)),
    ('KNN', KNeighborsClassifier(n_neighbors=1)),
    ('NB', GaussianNB()),
    ('RFC', RandomForestClassifier(random_state=42)),
    ('MLP', MLPClassifier(random_state=42, learning_rate='adaptive', max_iter=1000))
]

# whenever possible used joblib to speed-up the training
with joblib.parallel_backend('loky', n_jobs=-1):
    for label, clf in clfs:
        # train the model
        clf.fit(X_train, y_train)

        # generate predictions
        predictions = clf.predict(X_test)

        # compute the performance metrics
        mcc = matthews_corrcoef(y_test, predictions)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        print(f'{label:3} {acc:.2f} {f1:.2f} {mcc:.2f}')    
    
joblib.dump(clf, "model_header.pkl")

