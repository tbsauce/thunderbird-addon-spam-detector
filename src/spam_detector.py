import joblib
import datetime
from dateutil import parser
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText

# Ensure nltk resources are downloaded (can be commented out if already done)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text, lemmatizer, stop_words):
    # Check if the text is a string
    if not isinstance(text, str):
        return ""
    
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token).lower() for token in tokens if token.isalpha() and token.lower() not in stop_words and len(token) > 2])

def categorize_date(date_str):
    try:
        date_obj = parser.parse(date_str)
        day_start = datetime.time(9, 0, 0)
        day_end = datetime.time(18, 0, 0)

        if date_obj.weekday() < 5:
            return 0 if day_start <= date_obj.time() <= day_end else 1
        else:
            return 2 if day_start <= date_obj.time() <= day_end else 3
    except Exception as e:
        print(f"Error parsing date: {date_str} - {e}")
        return None

def extract_domain(email):
    if isinstance(email, str) and '@' in email:
        return email.split('@')[1].split('>')[0]
    return ''
def text_to_fasttext_features(text, model):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def preprocess_data(data, fasttext, enc):
    # Convert to DataFrame
    df = pd.DataFrame([data], columns=['Content', 'Subject', 'Sender', 'Return-Path', 'Reply-To', 'Date'])

    # Apply preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('portuguese'))
    df['Content'] = df['Content'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
    df['Subject'] = df['Subject'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
    df['Combined_Text'] = df['Subject'] + ' ' + df['Content']

    # Get FastText features
    df['FastText_Features'] = df['Combined_Text'].apply(lambda x: text_to_fasttext_features(x, fasttext))
    fasttext_features = np.stack(df['FastText_Features'].values)

    # Categorical features preprocessing
    df['Sender'] = df['Sender'].apply(extract_domain)
    df['Reply-To'] = df['Reply-To'].apply(extract_domain)
    df['Return-Path'] = df['Return-Path'].apply(extract_domain)
    df['Date'] = df['Date'].apply(categorize_date)

    # Encoding categorical features
    df[['Sender', 'Reply-To', 'Return-Path']] = df[['Sender', 'Reply-To', 'Return-Path']].astype('category')
    df[['Sender', 'Reply-To', 'Return-Path']] = df[['Sender', 'Reply-To', 'Return-Path']].apply(lambda x: x.cat.codes)

    # Apply OneHotEncoding
    encoded_data = pd.DataFrame(enc.transform(df[['Sender', 'Reply-To', 'Return-Path']]).toarray())

    # Combine all features
    df_final = pd.concat([df.drop(['Content', 'Subject', 'Combined_Text', 'FastText_Features'], axis=1), pd.DataFrame(fasttext_features), encoded_data], axis=1)
    df_final.columns = df_final.columns.astype(str)

    return df_final

def predict(data):
    # Load the trained models and transformers
    enc = joblib.load('hot_encoder.pkl')
    model = joblib.load('model.pkl')
    fasttext = joblib.load('fasttext.pkl')

    # Preprocess data
    processed_data = preprocess_data(data, fasttext, enc)

    # Predict
    processed_data = processed_data.values
    prediction = model.predict(processed_data)[0]
    return prediction

def main():

    # Example data (replace with actual email data for prediction)
    data = ['O que é que podes esperar? 1. Conversas com Consultores EF que te podem explicar como é estudar no estrangeiro e estar completamente imerso numa cultura nova. 2. A oportunidade que ganhar uma bolsa de estudo no estrangeiro de cerca de $5000. 3. Excursões virtuais dos campus EF .','Rainmaking is hiring: Graduate Software Engineer for Payton.','LinkedIn <jobs-listings@linkedin.com>','<s-4sebtqhpx3pa1707g7w34ggenuq16r330ya80329vcsqxatb05i8izat@bounce.linkedin.com>','',"Wed, 3 May 2023 17:33:21 +0000 (UTC)"]
    
    # Predict
    prediction = predict(data)

    # Display prediction result
    print("The message is classified as SPAM." if prediction == 1 else "The message is classified as NOT SPAM.")

if __name__ == "__main__":
    main()