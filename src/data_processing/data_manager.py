class DataManager:
    def __init__(
            self,
            accounts,
            concerts,
            planned_concerts,
            subscriptions,
            tickets,
            zipcodes):
        self.__accounts = accounts
        self.__concerts = concerts
        self.__planned_concerts = planned_concerts
        self.__subscriptions = subscriptions
        self.__tickets = tickets
        self.__zipcodes = zipcodes

    def accounts(self):
        return self.__accounts

    def concerts(self):
        return self.__concerts

    def planned_concerts(self):
        return self.__planned_concerts

    def subscriptions(self):
        return self.__subscriptions

    def tickets(self):
        return self.__tickets

    def zipcodes(self):
        return self.__zipcodes
