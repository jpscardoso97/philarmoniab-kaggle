import pandas as pd

class DataLoader:

    '''
    Load all data sources
    '''
    @staticmethod
    def load_data_sources():
        # Training data
        train = pd.read_csv('../data/train.csv')

        # Previously purchased subscriptions by account
        subscriptions = pd.read_csv('../data/subscriptions.csv')

        # Location info for each patron and donation history
        accounts = pd.read_csv('../data/account.csv')

        # Previous concerts by season
        concerts = pd.read_csv('../data/concerts.csv')

        # List of planned concert sets for the 2014-15 season
        planned_concerts = pd.read_csv('../data/concerts_2014-15.csv')

        # Previously purchased tickets by account
        tickets = pd.read_csv('../data/tickets_all.csv')

        # Location and demographic information for zipcodes
        zipcodes = pd.read_csv('../data/zipcodes.csv')

        # Final test data
        test = pd.read_csv('../data/test.csv')
        test['account.id'] = test['ID']
        test.drop('ID', axis=1, inplace=True)

        return train, subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes, test