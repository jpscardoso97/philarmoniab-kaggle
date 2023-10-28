import pandas as pd

class FeaturesBuilder:
    def __init__(self, data_manager):
        self.__data_manager = data_manager
    
    def build_features(self, X):
        #########################
        # Subscription features #
        #########################
        subscriptions = self.__data_manager.subscriptions().copy()
        subs_by_acc = subscriptions.groupby(['account.id'])
        subscriptions_by_account = pd.DataFrame({'num_subscriptions':subs_by_acc.size(), 
                                                'avg_seats_by_subscription': subs_by_acc['no.seats'].mean(),
                                                'sub_tier': subs_by_acc['subscription_tier'].apply(lambda x: x.mode().iloc[0]),
                                                'sub_seasons': subs_by_acc['season'].unique()
                                                }).reset_index()
        
        # Add average number of seats by subscription by account id (if can't find, then mean)
        X['avg_seats_by_subscription'] = X['account.id'].map(subscriptions_by_account.set_index('account.id')['avg_seats_by_subscription'])
        X['avg_seats_by_subscription'].fillna(subscriptions_by_account['avg_seats_by_subscription'].mean(), inplace=True)

        # Add average subscription price level by account id (if can't find, then mean)
        X['avg_subscription_price_level'] = X['account.id'].map(subs_by_acc['price.level'].mean())
        X['avg_subscription_price_level'].fillna(subscriptions['price.level'].mean(), inplace=True)

        # Add average subscription tier by account id (if can't find, then mean)
        X['avg_subscription_tier'] = X['account.id'].map(subs_by_acc['subscription_tier'].mean())
        X['avg_subscription_tier'].fillna(subscriptions['subscription_tier'].mean(), inplace=True)

        # Add median value for location by account id (if can't find, then mean)
        X['median_location_value'] = X['account.id'].map(subs_by_acc['location'].median())
        X['median_location_value'].fillna(int(subscriptions['location'].median()), inplace=True)

        # Let's create an aggregate feature that gives an estimate of how much the account spent on subscriptions
        # this value takes into consideration not only the amount of subscriptions but multiplies it by the price level of the subscription
        X['total_spent_in_subscriptions'] = X['account.id'].map(subs_by_acc['price.level'].sum())
        X['total_spent_in_subscriptions'].fillna(0, inplace=True)

        # Add average value for section by account id (if can't find, then mean)
        X['median_section_value'] = X['account.id'].map(subs_by_acc['section'].median())
        X['median_section_value'].fillna(int(subscriptions['section'].median()), inplace=True)

        # Add average value for season by account id (if can't find, then mean)
        X['avg_season_value'] = X['account.id'].map(subs_by_acc['season'].mean())
        X['avg_season_value'].fillna(int(subscriptions['season'].mean()), inplace=True)

        # Add average value for package by account id (if can't find, then mean)
        X['avg_package_value'] = X['account.id'].map(subs_by_acc['package'].mean())
        X['avg_package_value'].fillna(int(subscriptions['package'].mean()), inplace=True)

        ###################
        # Ticket features #
        ###################
        tickets = self.__data_manager.tickets().copy()
        tickets_by_acc = tickets.groupby(['account.id'])

        # Add average ticket price level by account id (if can't find, then mean)
        X['avg_ticket_price_level'] = X['account.id'].map(tickets_by_acc['price.level'].mean())
        X['avg_ticket_price_level'].fillna(tickets['price.level'].mean(), inplace=True)

        # Add total number of seats by account id (if can't find, then 0)
        X['num_seats'] = X['account.id'].map(tickets_by_acc['no.seats'].sum())
        X['num_seats'] = X['num_seats'].fillna(0)

        # Add total number of seats just for the 2013-2014 season by account id (if can't find, then 0)
        X['num_seats_2013'] = X['account.id'].map(tickets[tickets['season'] == '2013-2014'].groupby(['account.id'])['no.seats'].sum())
        X['num_seats_2013'] = X['num_seats_2013'].fillna(0)

        #####################
        # Concerts features #
        #####################
        concerts = self.__data_manager.concerts()
        planned_concerts = self.__data_manager.planned_concerts()

        # group concerts by season and aggregate list of unique conductors
        concerts['conductor'] = concerts['who'].apply(lambda x: x.split(',')[0])
        conductors_by_season = concerts.groupby(['season'])['conductor'].unique().reset_index()

        # create new column conductors in subscriptions_by_account with all the unique values as a flattened list from conductors_by_season where the season is one of the sub_seasons
        subscriptions_by_account['conductors'] = subscriptions_by_account['sub_seasons'].apply(lambda x: set([item for sublist in conductors_by_season[conductors_by_season['season'].isin(x)]['conductor'] for item in sublist]))

        # transform "who" column in planned_concerts to "conductors" column with just the name of the conductors
        planned_concerts['conductors'] = planned_concerts['who'].apply(lambda x: x.split(',')[0])
        
        # aggregate list of unique conductors in next season
        planned_conductors = planned_concerts['conductors'].unique()
        subscriptions_by_account['watched_conductors'] = subscriptions_by_account['conductors'].apply(lambda x: len(x.intersection(planned_conductors)))
        
        subscriptions_by_account.drop(['sub_seasons', 'conductors'], axis=1, inplace=True)

        X['watched_conductors'] = X['account.id'].map(subscriptions_by_account.set_index('account.id')['watched_conductors'])
        X['watched_conductors'] = X['watched_conductors'].fillna(0)

        ####################
        # Account features #
        ####################
        accounts = self.__data_manager.accounts()

        X = X.merge(accounts, on='account.id', how='left')
        X.drop(['no.donations.lifetime'], axis=1, inplace=True)

        return X
