
from sklearn.metrics import accuracy_score
from detector import predict
import pandas as pd

# Read the data from csv
data = pd.read_csv('./datasets/spam.csv', names=['label', 'message'])
#data = pd.read_csv('./datasets/Emails.csv', names=['message', 'label'])
#data = pd.read_csv('./datasets/messages.csv', names=['subject', 'message', 'label'])

# Preprocess and predict for each message in the dataset
predicted_labels = []
for index, row in data.iterrows():
    predicted_labels.append(predict(row['message']))
    
# Get the true labels from the dataset
true_labels = [1 if label == 'spam' else 0 for label in data['label']]
# true_labels = [1 if label == '1' else 0 for label in data['label']]

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")



