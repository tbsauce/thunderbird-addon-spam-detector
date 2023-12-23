import os
import email
import csv
import re  # Import regular expressions
from email.header import decode_header
import sys
from bs4 import BeautifulSoup

sys.path.append('..') 

def decode_subject(subject):
    decoded_subject, encoding = decode_header(subject)[0]
    if isinstance(decoded_subject, bytes):
        # Decode using the provided encoding, if available, or default to 'utf-8'
        return decoded_subject.decode(encoding if encoding else 'utf-8', errors='ignore')
    else:
        # Return the decoded string as is
        return decoded_subject

# Specify the folders containing .eml files for spam and non-spam
spam_folder_path = 'emails/spam'
non_spam_folder_path = 'emails/ham'

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
            subject = decode_subject(msg.get('subject',''))
            sender = msg.get('from','')
            return_path = msg.get('Return-Path', '')
            reply_to = msg.get('Reply-To','')
            date = msg.get('date','')

            # Extract the email content
            content = ""
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    content += part.get_payload(decode=True).decode("utf-8", errors="ignore")
                elif content_type == "text/html" and not content:
                    html_content = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html_content, 'html.parser')
                    content += soup.get_text()

            # Remove excessive whitespace from the content
            content = re.sub(r'\s+', ' ', content).strip()

            # Append email information, content, and label to the list
            email_data.append([label, str(content), str(subject), sender, return_path ,reply_to, date])

# Process the spam folder (label 1)
process_folder(spam_folder_path, 1)

# Process the non-spam folder (label 0)
process_folder(non_spam_folder_path, 0)

# Specify the CSV file path where you want to save the data
csv_file_path = 'training_data.csv'

# Write email data to the CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers, including the new "Label" header
    csv_writer.writerow(['Label','Content', 'Subject', 'Sender', 'Return-Path' , 'Reply-To', 'Date'])
    
    # Write email data
    csv_writer.writerows(email_data)

print(f'{len(email_data)} emails have been saved to {csv_file_path}')
