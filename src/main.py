import sys
import pickle
from os import path

from data_processing.data_loader import DataLoader
from data_processing.data_cleaner import DataCleaner
from data_processing.data_manager import DataManager
from training.rf_trainer import RandomForestTrainer
from testing.model_runner import ModelRunner
from features.features_builder import FeaturesBuilder

sys.path.append('data_processing')
sys.path.append('features')
sys.path.append('training')
sys.path.append('testing')

def main():
    model_path = '../saved_models/rf_model.sav'

    # Load data
    print("Loading data...")
    train_data, subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes, test_data = DataLoader.load_data_sources()

    # Clean data
    print("Cleaning data...")
    cleaner = DataCleaner()
    subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes = cleaner.clean(subscriptions, accounts, concerts, planned_concerts, tickets, zipcodes)

    # Create data manager
    print("Creating data manager...")
    data_manager = DataManager(accounts, concerts, planned_concerts, subscriptions, tickets, zipcodes)

    # Create feature builder
    features_builder = FeaturesBuilder(data_manager)

    # Build features and train model if needed

    # Check trained model file exists
    if path.exists(model_path):
        print("Using saved model...")
        model = pickle.load(open(model_path, 'rb'))
    else:
        # Train model
        print("Training model...")
        trainer = RandomForestTrainer(features_builder)
        model = trainer.train(train_data)

    # Save trained model
    print("Saving model...")
    pickle.dump(model, open(model_path, 'wb'))

    # Make predictions, output results and generate predicions file
    print("Making predictions...")
    runner = ModelRunner(features_builder)
    runner.test(model, test_data)

main()