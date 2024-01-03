import joblib
import datetime
from dateutil import parser
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText
import random
import warnings

# Ignore the specific UserWarning about feature names in RandomForestClassifier
warnings.filterwarnings("ignore", message="X has feature names, but RandomForestClassifier was fitted without feature names", category=UserWarning)


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

    
    # Categorical features preprocessing
    df['Sender'] = df['Sender'].apply(extract_domain)
    df['Reply-To'] = df['Reply-To'].apply(extract_domain)
    df['Return-Path'] = df['Return-Path'].apply(extract_domain)
    df['Sender-Return'] = df.apply(lambda row: 1 if row['Sender'] == row['Return-Path'] else 0, axis=1)
    
    df['Date'] = df['Date'].apply(categorize_date)
    df['Day'] = df['Date'].apply(lambda x: 1 if x in [0, 2] else 0)
    df['Weekday'] = df['Date'].apply(lambda x: 1 if x in [1, 4] else 0)


    # Encoding categorical features
    df[['Sender', 'Reply-To', 'Return-Path']] = df[['Sender', 'Reply-To', 'Return-Path']].astype('category')

    # Apply OneHotEncoding
    encoded_data = pd.DataFrame(enc.transform(df[['Sender', 'Reply-To', 'Return-Path']]))
    encoded_data = encoded_data.add_prefix('encoded_')

    # Get FastText features
    df['FastText_Features'] = df['Combined_Text'].apply(lambda x: text_to_fasttext_features(x, fasttext))
    fasttext_features = np.stack(df['FastText_Features'].values)
    fasttext_features = pd.DataFrame(fasttext_features)
    fasttext_features = fasttext_features.add_prefix('fasttext_')

    # Combine all features
    df_final = pd.concat([df.drop(['Content', 'Subject', 'Sender', 'Reply-To', 'Return-Path', 'Date','Combined_Text', 'FastText_Features'], axis=1), encoded_data, fasttext_features], axis=1)
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
    #processed_data = processed_data.values
    prediction = model.predict(processed_data)[0]
    return prediction
def main():
    # Read the test data from CSV into a DataFrame
    test_data = pd.read_csv('training/test_data.csv')  # Change the filename as per your test data file

    # Number of lines in the test data
    num_lines = len(test_data)

    # Initialize correct predictions counter
    correct_predictions = 0

    # Perform predictions for each row in the test data
    for index, row in test_data.iterrows():
        # Extract data for prediction from the row
        data = [row['Content'], row['Subject'], row['Sender'], row['Return-Path'], row['Reply-To'], row['Date']]
        
        # Make a prediction using your predict() function (replace this with your prediction logic)
        prediction = predict(data)  # Implement your prediction function
        
        # Check if the prediction matches the actual label
        if prediction == row['Label']:
            correct_predictions += 1

        # Display accuracy after every 10 iterations
        if (index + 1) % 10 == 0:  # Check if 10 iterations have passed
            accuracy = correct_predictions / (index + 1) * 100
            print(f"Iteration {index + 1}: Accuracy: {accuracy:.2f}%")

    # Calculate accuracy after all iterations
    accuracy = correct_predictions / num_lines * 100

    # Display the final accuracy
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()