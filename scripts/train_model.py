# scripts/train_model.py

import logging
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib;

def train_model():
    # Load dataset
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

#    Save the trained model to model.pkl
    joblib.dump(model, 'model/model.pkl')
    # Log information
    logging.info("Model training complete.")

if __name__ == "__main__":
    train_model()
