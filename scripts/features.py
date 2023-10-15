import pandas as pd
import numpy as np

def get_features(df, accounts, subscriptions):
    # Add features to the training data
    ## Account features
    df['ID'] = df['account.id']
    df = pd.merge(df, accounts, left_on='ID', right_on='account.id', how='left')
    
    # Subscription features
    subscriptions_by_account = pd.DataFrame({'num_subscriptions':subscriptions.groupby(['account.id']).size()}).reset_index()
    # Add total subscriptions by account id (if can't find, then 0)
    df['num_subscriptions'] = df['ID'].map(subscriptions_by_account.set_index('account.id')['num_subscriptions'])
    # Assuming that if account doesn't appear on subscriptions data, then they have no subscriptions
    df['num_subscriptions'] = df['num_subscriptions'].fillna(0)

    return df

# Account data
def clean_account_data(df):
    df = clean_locations(df)
    df = drop_unnecessary_columns(df)

    return df

def clean_locations(df):
    # check rows that have either a shipping.zip.code or shipping.citys
    print("There are {} rows that have NaN zip code and city name".format(len(df.loc[df['shipping.zip.code'].isnull() & df['shipping.city'].isnull()])))

    return df

def drop_unnecessary_columns(df):
    df = df.drop(columns=['shipping.zip.code', 'shipping.city', 'relationship', 'first.donated'])

    return df