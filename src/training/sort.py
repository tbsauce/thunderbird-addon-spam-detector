import os
import email
import csv
from email.header import decode_header
import sys
sys.path.append('..') 
from content_detector import predict


# Function to decode email subject
def decode_subject(subject):
    decoded_subject = decode_header(subject)[0]
    return decoded_subject[0] if decoded_subject[0] else subject

# Specify the folders containing .eml files for spam and non-spam
spam_folder_path = '/home/sauce/thunderbird-addon-spam-detector/src/training/spam'
non_spam_folder_path = '/home/sauce/thunderbird-addon-spam-detector/src/training/normal'

# Create a list to store email information
email_data = []

# Function to process a folder and assign labels
def process_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        if filename.endswith('.eml'):
            eml_file_path = os.path.join(folder_path, filename)
            
            # Parse the .eml file
            with open(eml_file_path, 'rb') as eml_file:
                msg = email.message_from_binary_file(eml_file)
            
            # Extract relevant email information (e.g., subject, sender, date)
            subject = decode_subject(msg.get('subject', 'No Subject'))
            sender = msg.get('from', 'No Sender')
            to = msg.get('to', 'No Recipient')
            reply_to = msg.get('Reply-To', 'Nobody to reply')
            date = msg.get('date', 'No Date')

            # Extract the email content
            content = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    content += part.get_payload(decode=True).decode("utf-8", errors="ignore")

            # Append email information, content, and label to the list
            email_data.append([label, predict(content), predict(str(subject)), sender, to, reply_to, date])

# Process the spam folder (label 1)
process_folder(spam_folder_path, 1)

# Process the non-spam folder (label 0)
process_folder(non_spam_folder_path, 0)

# Specify the CSV file path where you want to save the data
csv_file_path = '/home/sauce/thunderbird-addon-spam-detector/src/datasets/subpredicted.csv'

# Write email data to the CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers, including the new "Label" header
    csv_writer.writerow(['Label','Content', 'Subject', 'Sender', 'To', 'Reply-To', 'Date'])
    
    # Write email data
    csv_writer.writerows(email_data)

print(f'{len(email_data)} emails have been saved to {csv_file_path}')
