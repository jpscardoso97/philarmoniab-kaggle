import pandas as pd

from sklearn.preprocessing import LabelEncoder

class DataCleaner:
    def __init__(self):
        self.__label_encoder = LabelEncoder()
    
    def clean(self, subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes):

        #################
        # Subscriptions #
        #################

        initial_len = len(subscriptions)
        subscriptions = subscriptions.dropna()
        print("Removed ", initial_len-len(subscriptions), " subscriptions after cleaning")
        
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
        subscriptions['season'] = self.__label_encoder.fit_transform(subscriptions['season'])
        subscriptions['package'] = self.__label_encoder.fit_transform(subscriptions['package'])
        subscriptions['location'] = self.__label_encoder.fit_transform(subscriptions['location'])
        subscriptions['multiple.subs'] = self.__label_encoder.fit_transform(subscriptions['multiple.subs'])
   
        print("Subscription categorical data encoded...")

        ############
        # Accounts #
        ############
        initial_len = len(accounts)
        
        # Keep only relevant columns on accounts and convert full date of first donation to year of donation
        accounts = accounts[['account.id', 'amount.donated.2013', 'amount.donated.lifetime', 'no.donations.lifetime', 'first.donated', 'billing.zip.code']]
        accounts['first.donated'] = pd.to_datetime(accounts['first.donated']).dt.year
        accounts['first.donated'].fillna(0, inplace=True)

        # Encode billing.zip.code column
        accounts['billing.zip.code'] = self.__label_encoder.fit_transform(accounts['billing.zip.code'])
        accounts['billing.zip.code'].fillna("", inplace=True)

        accounts = accounts.dropna()

        print("Removed ", initial_len-len(accounts), " accounts after cleaning")

        ############
        # Concerts #
        ############
        initial_len = len(concerts)
        concerts = concerts[["season", "who"]]
        concerts = concerts.dropna()

        print("Removed ", initial_len-len(concerts), " concerts after cleaning")

        ###########
        # Tickets #
        ###########
        tickets = tickets[['account.id', 'price.level', 'no.seats', 'season']]

        tickets['price.level'] = self.__label_encoder.fit_transform(tickets['price.level'])

        initial_len = len(tickets)
        tickets = tickets.dropna()

        print("Removed ", initial_len-len(tickets), " tickets after cleaning")

        ############
        # Zipcodes #
        ############
        zipcodes = zipcodes[['Zipcode', 'City', 'Lat', 'Long', 'TotalWages', 'EstimatedPopulation']]

        initial_len = len(zipcodes)

        print("Removed ", initial_len-len(zipcodes), " zipcodes after cleaning")

        zipcodes['City'] = self.__label_encoder.fit_transform(zipcodes['City'])
        zipcodes = zipcodes.dropna()

        return subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes
