import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_lexical_features

DATA_PATH = 'data/PhiUSIIL.csv' # Check your actual filename space
SCALER_PATH = 'models/scaler.joblib'
XGB_MODEL_PATH = 'models/xgb_model.json'
MLP_MODEL_PATH = 'models/mlp_model.keras'

def build_mlp_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    os.makedirs('models', exist_ok=True)

    print(f"Loading dataset from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        print("Columns in dataset:", df.columns)
    except FileNotFoundError:
        print("Error: Dataset not found. Please run download_data.py first.")
        return

    print("Extracting lexical features for training...")
    # We ONLY use lexical features now. Network features are heuristics in server.py
    feature_dicts = []
    # Using a subset for demonstration speed if needed, remove .head() for full train
    for url in df['URL']: 
        # Ensure URL is string
        feature_dicts.append(extract_lexical_features(str(url)))
    
    features_df = pd.DataFrame(feature_dicts)
    
    # Labels
    y = df['status'].values

    print(f"Feature set shape: {features_df.shape}")
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 1. Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    val_acc_xgb = xgb_model.score(X_val, y_val)
    print(f"XGBoost Validation Accuracy: {val_acc_xgb:.4f}")

    # 2. Train MLP
    print("Training MLP...")
    mlp_model = build_mlp_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    mlp_model.fit(
        X_train, y_train, 
        epochs=30, 
        batch_size=64, 
        validation_data=(X_val, y_val), 
        callbacks=[early_stop], 
        verbose=1
    )

    # Save Artifacts
    print("Saving artifacts...")
    xgb_model.save_model(XGB_MODEL_PATH)
    mlp_model.save(MLP_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Done! System is ready for server.py")

if __name__ == '__main__':
    main()