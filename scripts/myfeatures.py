import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#numerical_features = ['amount.donated.2013', 'amount.donated.lifetime', 'no.donations.lifetime', 'num_subscriptions']

def build_features(df, accounts, subscriptions, tickets):
    # Add features to the training data
    ## Account features
    # check if column exists
    if 'ID' not in df.columns:
        df['ID'] = df['account.id']

    # Subscription features
    subscriptions_by_account = pd.DataFrame({'num_subscriptions':subscriptions.groupby(by=['account.id']).size()}).reset_index()
    # Add total subscriptions by account id (if can't find, then 0)
    df['num_subscriptions'] = df['ID'].map(subscriptions_by_account.set_index('account.id')['num_subscriptions'])
    # Assuming that if account doesn't appear on subscriptions data, then they have no subscriptions
    df['num_subscriptions'] = df['num_subscriptions'].fillna(0)

    # Add average subscription price level by account id (if can't find, then 0)
    df['avg_subscription_price_level'] = df['ID'].map(subscriptions.groupby(['account.id'])['price.level'].mean())
    df['avg_subscription_price_level'] = df['avg_subscription_price_level'].fillna(0)

    # Add average subscription tier by account id (if can't find, then 0)
    df['avg_subscription_tier'] = df['ID'].map(subscriptions.groupby(['account.id'])['subscription_tier'].mean())
    df['avg_subscription_tier'] = df['avg_subscription_tier'].fillna(0)

    # Ticket features
    tickets_by_account = pd.DataFrame({'num_tickets':tickets.groupby(['account.id']).size()}).reset_index()
    # Add total tickets by account id (if can't find, then 0)
    df['num_tickets'] = df['ID'].map(tickets_by_account.set_index('account.id')['num_tickets'])
    # Assuming that if account doesn't appear on tickets data, then they have no tickets
    df['num_tickets'] = df['num_tickets'].fillna(0)

    # Add total tickets just for the 2013-2014 season by account id (if can't find, then 0)
    df['num_tickets_2013'] = df['ID'].map(tickets[tickets['season'] == '2013-2014'].groupby(['account.id'])['no.seats'].sum())
    df['num_tickets_2013'] = df['num_tickets_2013'].fillna(0)

    # Add average ticket price level by account id (if can't find, then 0)
    df['avg_ticket_price_level'] = df['ID'].map(tickets.groupby(['account.id'])['price.level'].mean())
    df['avg_ticket_price_level'] = df['avg_ticket_price_level'].fillna(0)

    # Add total number of seats by account id (if can't find, then 0)
    df['num_seats'] = df['ID'].map(tickets.groupby(['account.id'])['no.seats'].sum())
    df['num_seats'] = df['num_seats'].fillna(0)

    # Add total number of seats just for the 2013-2014 season by account id (if can't find, then 0)
    df['num_seats_2013'] = df['ID'].map(tickets[tickets['season'] == '2013-2014'].groupby(['account.id'])['no.seats'].sum())
    df['num_seats_2013'] = df['num_seats_2013'].fillna(0)

    # Account features
    df = pd.merge(df, accounts, left_on='ID', right_on='account.id', how='left')
    #print("Missing account data: ", df.isnull().sum())

    df = clean_account_data(df)

    return df

# Account data
def clean_account_data(df):
    df = df.drop(columns=['billing.zip.code', 'billing.city','shipping.zip.code', 'shipping.city', 'relationship', 'first.donated'])

    return df

# Hot encode categorical variables
def hot_encode(df, columns):
    for column in columns:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df = df.drop(columns=[column])

    return df

def add_conductor_feature(df, subscriptions, concerts, planned_concerts):
    subs_by_acc = subscriptions.groupby(['account.id'])

    subscriptions_by_account = pd.DataFrame({'num_subscriptions':subs_by_acc.size(), 
                                            'sub_tier': subs_by_acc['subscription_tier'].apply(lambda x: x.mode().iloc[0]),
                                            'sub_seasons': subs_by_acc['season'].unique()
                                            }).reset_index()
    
    # group concerts by season and aggregate list of unique conductors
    concerts['conductor'] = concerts['who'].apply(lambda x: x.split(',')[0])
    conductors_by_season = concerts.groupby(['season'])['conductor'].unique().reset_index()
    
    # create new column conductors in subscriptions_by_account with all the unique values as a flattened list from conductors_by_season where the season is one of the sub_seasons
    subscriptions_by_account['conductors'] = subscriptions_by_account['sub_seasons'].apply(lambda x: set([item for sublist in conductors_by_season[conductors_by_season['season'].isin(x)]['conductor'] for item in sublist]))
    
    # transform "who" column in planned_concerts to "conductors" column with just the name of the conductors
    planned_concerts['conductors'] = planned_concerts['who'].apply(lambda x: x.split(',')[0])
    #display(planned_concerts.head())

    # aggregate list of unique conductors in next season
    planned_conductors = planned_concerts['conductors'].unique()

    subscriptions_by_account['watched_conductors'] = subscriptions_by_account['conductors'].apply(lambda x: len(x.intersection(planned_conductors)))

    #display(planned_conductors)
    #display(conductors_by_season)           
    #display(subscriptions_by_account.iloc[0]['conductors'])
    #display(subscriptions_by_account['watched_conductors'].value_counts())
    #display(subscriptions_by_account[subscriptions_by_account['conductors'].apply(lambda x: len(x)) == 8])

    subscriptions_by_account.drop(['sub_seasons', 'conductors'], axis=1, inplace=True)

    df['watched_conductors'] = df['account.id'].map(subscriptions_by_account.set_index('account.id')['watched_conductors'])
    df['watched_conductors'] = df['watched_conductors'].fillna(0)

    return df