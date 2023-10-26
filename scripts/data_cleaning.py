import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes):
    label_encoder = LabelEncoder()

    print("Missing subscriptions data: ", subscriptions.isnull().sum(), "out of", len(subscriptions), "\n")
    print("Dropping rows with missing data...")
    print("Initial length of subscriptions: ", len(subscriptions))
    subscriptions = subscriptions.dropna()
    print("Length of subscriptions after cleaning: ", len(subscriptions))

    print("Encoding subscriptions categorical columns...")
    
    sections_map = {
    'Balcony Rear': 1, 
    'Balcony': 2,
    'Boxes House Right': 3, 
    'Boxes House Left': 4, 
    'Gallery': 5, 
    'Floor': 6, 
    'Balcony Front': 7, 
    'Dress Circle': 8, 
    'Orchestra Rear': 9, 
    'Santa Rosa': 10, 
    'Orchestra': 11, 
    'Orchestra Front': 12, 
    'Premium Orchestra': 13, 
    'Box (Most Expensive)': 14}

    subscriptions['section'] = subscriptions['section'].map(sections_map)

    print("Missing accounts data: ", accounts.isnull().sum(), "out of", len(accounts), "\n")
    print("Dropping rows with missing data...")
    print("Initial length of accounts: ", len(accounts))
    
    # Keep only amount.donated.2013,amount.donated.lifetime,no.donations.lifetime,first.donated columns on accounts
    accounts = accounts[['account.id', 'amount.donated.2013', 'amount.donated.lifetime', 'no.donations.lifetime', 'first.donated', 'billing.zip.code']]
    accounts['first.donated'] = pd.to_datetime(accounts['first.donated'])
    accounts = accounts.dropna()
    print("Length of accounts after cleaning: ", len(accounts))

    print("Missing concerts data: ", concerts.isnull().sum(), "out of", len(concerts), "\n")
    print("Dropping rows with missing data...")
    print("Initial length of concerts: ", len(concerts))
    concerts = concerts.dropna()
    print("Length of concerts after cleaning: ", len(concerts))

    print("Missing planned concerts data: ", planned_concerts.isnull().sum(), "out of", len(planned_concerts), "\n")

    print("Missing tickets data: ", tickets.isnull().sum(), "out of", len(tickets), "\n")
    print("Keeping only relevant columns...")
    tickets = tickets[['account.id', 'price.level', 'no.seats', 'season']]

    print("Encoding price.level column")
    label_encoder = LabelEncoder()
    tickets['price.level'] = label_encoder.fit_transform(tickets['price.level'])

    print("Dropping rows with missing data...")
    print("Initial length of tickets: ", len(tickets))
    tickets = tickets.dropna()
    print("Length of tickets after cleaning: ", len(tickets))

    print("Missing zipcodes data: ", zipcodes.isnull().sum(), "out of", len(zipcodes), "\n")
    zipcodes = zipcodes.dropna()
    zipcodes['Zipcode'] = zipcodes['Zipcode'].astype(str)

    return subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes