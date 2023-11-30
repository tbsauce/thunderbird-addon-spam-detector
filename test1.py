import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the model and vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Read the data from spam.csv
data = pd.read_csv('spam1.csv', names=['label', 'message'])

# Preprocess and predict for each message in the dataset
predicted_labels = []
for index, row in data.iterrows():
    transformed_message = transform_text(row['message'])
    vector_input = tfidf.transform([transformed_message])
    prediction = model.predict(vector_input)[0]
    predicted_labels.append(prediction)

# Get the true labels from the dataset
true_labels = [1 if label == 'spam' else 0 for label in data['label']]

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

