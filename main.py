import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = 'data/water_potability.csv'
SCREENSHOT_DIR = 'screenshots'

# Ensure screenshot directory exists
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

def load_data(path):
    """
    Loads the dataset from the CSV file.
    """
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found at {path}. Please check the path.")
        return None

def preprocess_data(df):
    """
    Handles missing values and separates Features (X) and Target (y).
    Strategy: Fill missing values with the mean of the column.
    """
    print("[INFO] Preprocessing data...")
    
    # Check for null values
    if df.isnull().sum().sum() > 0:
        print(f"   - Found {df.isnull().sum().sum()} missing values. Imputing with mean.")
        df.fillna(df.mean(), inplace=True)
    
    # Split into Features (X) and Target (y)
    # 'Potability' is the target column (0 = Unsafe, 1 = Safe)
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    return X, y

def generate_visualizations(df):
    """
    Generates correlation heatmap and distribution plots.
    Saves them to the 'screenshots' folder for the report.
    """
    print("[INFO] Generating visualizations...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{SCREENSHOT_DIR}/correlation_heatmap.png")
    plt.close()
    print(f"   - Saved correlation_heatmap.png to {SCREENSHOT_DIR}")

    # 2. Potability Count Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Potability', data=df, palette='viridis')
    plt.title("Distribution of Potable (1) vs Not Potable (0) Water")
    plt.savefig(f"{SCREENSHOT_DIR}/class_balance.png")
    plt.close()

def train_model(X, y):
    """
    Trains a Random Forest Classifier.
    Returns the model, scaler, and accuracy metrics.
    """
    print("[INFO] Training Model...")
    
    # Split data: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling data (Important for ML performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{SCREENSHOT_DIR}/confusion_matrix.png")
    plt.close()
    
    return model, scaler

def predict_sample(model, scaler):
    """
    Simulates a real-world test with custom input.
    """
    print("\n[DEMO] Predicting specific water sample...")
    
    # Example Data: [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    # This is a high-quality water sample example
    sample_data = np.array([[7.2, 200.0, 20000.0, 7.5, 330.0, 450.0, 14.0, 60.0, 3.5]])
    
    # Scale the input
    sample_scaled = scaler.transform(sample_data)
    
    # Predict
    prediction = model.predict(sample_scaled)
    result = "Safe to Drink (Potable)" if prediction[0] == 1 else "Unsafe for Consumption"
    
    print(f"   Input Sample: {sample_data}")
    print(f"   Prediction: {result}")

if __name__ == "__main__":
    # 1. Load
    df = load_data(DATA_PATH)
    
    if df is not None:
        # 2. Visualize
        generate_visualizations(df)
        
        # 3. Preprocess
        X, y = preprocess_data(df)
        
        # 4. Train & Evaluate
        trained_model, trained_scaler = train_model(X, y)
        
        # 5. Demo Prediction
        predict_sample(trained_model, trained_scaler)
        
        print("\n[SUCCESS] Process completed. Check 'screenshots' folder for graphs.")
