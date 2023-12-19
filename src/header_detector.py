import joblib
import datetime
from dateutil import parser
import pandas as pd

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

def preprocess_data(data):

    # Convert to DataFrame
    df = pd.DataFrame([data], columns=['Content', 'Subject', 'Sender', 'Return-Path' ,'Reply-To', 'Date'])
    
    # Apply the function to extract domains
    df['Return-Path_Domain'] = df['Return-Path'].apply(extract_domain)
    df['Sender_Domain'] = df['Sender'].apply(extract_domain)
    # Compare and assign values
    df['Similar-Return'] = df.apply(lambda row: 1 if row['Return-Path_Domain'] == row['Sender_Domain'] else 0, axis=1)

    # Optionally, you can remove the temporary columns if they are not needed
    df.drop(['Return-Path_Domain', 'Sender_Domain'], axis=1, inplace=True)

    # Extracting domains from email addresses
    df['Sender'] = df['Sender'].apply(extract_domain)
    df['Reply-To'] = df['Reply-To'].apply(extract_domain)
    df['Return-Path'] = df['Return-Path'].apply(extract_domain)

    df['Date'] = df['Date'].apply(categorize_date)

    # Load OneHotEncoder
    enc = joblib.load('/home/sauce/thunderbird-addon-spam-detector/src/encoder.pkl')

    print(df.head())

    df['Sender'] = df['Sender'].astype('category')
    df['Reply-To'] = df['Reply-To'].astype('category')
    df['Return-Path'] = df['Return-Path'].astype('category')

    df['Sender'] = df['Sender'].cat.codes
    df['Reply-To'] = df['Reply-To'].cat.codes
    df['Return-Path'] = df['Return-Path'].cat.codes
    # Apply OneHotEncoding
    encoded_data = pd.DataFrame(enc.transform(df[['Sender', 'Reply-To', 'Return-Path']]).toarray())
    
    # Combine with original data
    df = df.join(encoded_data)

    return df

def predict(data):
    # Preprocess data
    processed_data = preprocess_data(data)

    processed_data.columns = processed_data.columns.astype(str)

    print(processed_data)
    # Load model and predict
    with open('/home/sauce/thunderbird-addon-spam-detector/src/model_header.pkl', 'rb') as model_file:
        model = joblib.load(model_file)

    prediction = model.predict(processed_data)[0]
    return prediction

def main():
    # Define the message to classify
    #data = [0,0,"LinkedIn Job Alerts <jobalerts-noreply@linkedin.com>","<s-2mit8lgg81b8h89cfql1l42btb06fag1ggoj10gm7cmtqrult0j81202@bounce.linkedin.com>","","Mon, 4 Dec 2023 21:24:23 +0000 (UTC)"]
    data = [1,0,'Tracktion <tracktioneer@tracktion.com>','<0101018c04697dd6-9e1ca011-1215-43f6-9cae-bbb6fe54f7a7-000000@mail4.tracktion.com>','Tracktion <tracktioneer@tracktion.com>',"Sat, 25 Nov 2023 02:57:51 +0000"]
    #data_non = [0,0,'ISIC News <news@isic.pt>','<bounce+5a72d6.137256-TELMOBELASAUCE=gmail.com@isic.pt>','ISIC <news@isic.pt>',"Tue, 12 Sep 2023 01:55:02 +0000"]
    prediction = predict(data)
    # Display prediction result
    if prediction == 1:
        print("The message is classified as SPAM.")
    else:
        print("The message is classified as NOT SPAM.")

if __name__ == "__main__":
    main()
