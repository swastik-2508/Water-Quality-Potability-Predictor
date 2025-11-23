import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class WaterQualityModel:
    """
    A class to encapsulate the Water Quality Machine Learning logic.
    This handles data cleaning, training, and prediction.
    """

    def __init__(self):
        # Initialize the algorithm and the scaler
        # n_estimators=100 means we use 100 decision trees
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean') # logic to fill missing values
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def load_and_clean(self, filepath):
        """
        Logic to load CSV and handle null values.
        """
        try:
            df = pd.read_csv(filepath)
            
            # Separate Features and Target
            X = df.drop('Potability', axis=1)
            y = df['Potability']
            
            # Logic: Handle Missing Values using Mean Imputation
            # We fit on X to learn the means
            X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            
            return X_imputed, y, df # Return df for visualization purposes
        except Exception as e:
            print(f"[ERROR] Logic Failure in loading data: {e}")
            return None, None, None

    def train(self, X, y):
        """
        Logic to split data, scale features, and train the model.
        """
        # Split logic: 80% Train, 20% Test
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling logic: Normalize data to have Mean=0, Variance=1
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Training logic
        self.model.fit(X_train_scaled, y_train)
        
        # Generate predictions for internal evaluation
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        return accuracy_score(self.y_test, self.y_pred)

    def get_evaluation_metrics(self):
        """
        Returns the confusion matrix and classification report.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        return cm, report

    def predict_new_sample(self, input_features):
        """
        Logic to predict a single new water sample.
        Input: List of 9 float values
        Output: "Potable" or "Not Potable"
        """
        # Convert list to numpy array and reshape
        data_array = np.array(input_features).reshape(1, -1)
        
        # Apply the SAME scaling used in training
        scaled_data = self.scaler.transform(data_array)
        
        # Predict
        prediction = self.model.predict(scaled_data)
        
        return "Safe (Potable)" if prediction[0] == 1 else "Unsafe (Not Potable)"
