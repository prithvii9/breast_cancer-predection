
#############################################################################################################
# Required Python Packages
#############################################################################################################
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#############################################################################################################
# File Paths
#############################################################################################################
OUTPUT_PATH = 'breast-cancer-wisconsin.csv'
MODEL_PATH = "bc_model_pipeline.joblib"


#############################################################################################################
# Headers
#############################################################################################################
HEADERS = ['CodeNumber', 'ClumpThickness', 'UniformityCellSize',
       'UniformityCellShape', 'MarginalAdhesion', 'SingleEpithelialCellSize',
       'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses',
       'CancerType']

#############################################################################################################
# Function name :       dataset_statistics
# Description :         Display statistics
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def dataset_statistics(dataset):
    # Print basic stats
    print(dataset.describe())


#############################################################################################################
# Function name :       handle_missing_values
# Description :         Filter missing values from the dataset
# Input :               Dataset with missing values
# Output :              Dataset by removing the missing values
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def handle_missing_values(dataset, feature_headers):
    """
    Convert '?' to np.NaN and let Simple Imputer handle them inside Pipeline.
    Keep only numeric columns in features 
    """
    # Replace '?' in whole dataframe
    dataset = dataset.replace('?', np.nan)

    # Cast features to numeric
    dataset[feature_headers] = dataset[feature_headers].apply(pd.to_numeric, errors = 'coerce')

    return dataset
    

#############################################################################################################
# Function name :       split_dataset
# Description :         Split the dataset with train_percentage
# Input :               Dataset with related information
# Output :              Dataset after splitting
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def split_dataset(df, features, target, train_percentage):
    X_train, X_test, Y_train, Y_test = train_test_split(df[features], df[target], train_size=train_percentage, random_state=42)

    return X_train, X_test, Y_train, Y_test


#############################################################################################################
# Function name :       build_pipeline
# Description :         Build a pipeline
# Simple Imputer :      replace missing(NaN values) with median
# Random Forest :       robust baseline         
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def build_pipeline():
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight=None
            ))
    ])

    return pipe


#############################################################################################################
# Function name :       train_pipeline
# Description :         Train a pipeline
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def train_pipeline(pipe, X_train, Y_train):
    pipe.fit(X_train, Y_train)
    return pipe



#############################################################################################################
# Function name :       plot_feature_importances
# Description :         Display the feature importance
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def plot_feature_importances(model, features_names, title="Feature Importances (Random Forest)"):
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Feature importance not available for this model")
        return
    
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,6))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), [features_names[i] for i in idx], rotation=45, ha = 'right')
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()




#############################################################################################################
# Function name :       save_model
# Description :         Save the model
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved to {path}")



#############################################################################################################
# Function name :       load_model
# Description :         Load the trained model
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from path : {path}")
    return model




#############################################################################################################
# Function name :       main
# Description :         Main function from where execution starts
# Author :              Prithviraj Chavan
# Date :                10/08/2025
#############################################################################################################
def main():
    # 1) Load CSV
    dataset = pd.read_csv(OUTPUT_PATH)

    # 2) Drop unnecessary columns 
    dataset.drop(columns=HEADERS[0], inplace=True)

    # 3) Basic stats
    dataset_statistics(dataset)

    # 4) Prepare features/target
    feature_headers = HEADERS[1:-1]     # Drop CodeNumber, keep all features
    target_headers = HEADERS[-1]        # CancerType (benign = 2, maligant = 4)

    # 5) Handle '?' and coerce to numeric; imputation will happen inside Pipeline
    dataset = handle_missing_values(dataset, feature_headers)

    # 6) Split
    X_train, X_test, Y_train, Y_test = split_dataset(dataset, feature_headers, target_headers, 0.7)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"Y_train shape : {Y_train.shape}")
    print(f"Y_test shape : {Y_test.shape}")

    # 7) Build + Train Pipeline
    pipeline = build_pipeline()
    trained_model = train_pipeline(pipeline, X_train, Y_train)


    # 8)Predictions
    predictions = trained_model.predict(X_test)

    # 9) Metrics
    print(f"Training Accuracy : {accuracy_score(Y_train, trained_model.predict(X_train))}")
    print(f"Testing Accuracy : {accuracy_score(Y_test, predictions)}")
    print(f"Classification Report :\n{classification_report(Y_test, predictions)}")
    print(f"Confusion Matrix :\n {confusion_matrix(Y_test, predictions)}")

    # 10) Feature importances(tree-based)
    plot_feature_importances(trained_model, feature_headers, title="Feature Importances (RF)")

    # 11) Save model (Pipeline) using joblib
    save_model(trained_model, MODEL_PATH)

    # 12) Load model and test a sample
    loaded_model = load_model(MODEL_PATH)
    sample = X_test.iloc[[0]]
    pre_loaded = loaded_model.predict(sample)
    print(f"Loaded model prediction for sample : {pre_loaded[0]}")


#############################################################################################################
# Application Starter
#############################################################################################################
if __name__ == "__main__":
    main()