import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class WaterQualityModel:
    """
    Encapsulates the Machine Learning logic for the AquaSafe project.
    """
    def _init_(self):
     # Using Random Forest as it handles non-linear relationships well
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def load_and_clean(self, filepath):
        """
        Loads CSV, handles missing values, and separates features/target.
        """
        try:
            df = pd.read_csv(filepath)
    
            # Features: All columns except 'Potability'
            # Target: 'Potability'
            X = df.drop('Potability', axis=1)
            y = df['Potability']
        
            # Impute missing values (NaN) with the mean of the column
            X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
            return X_imputed, y, df
        except Exception as e:
            print(f"[ERROR] Could not process data: {e}")
            return None, None, None



    def train(self, X, y):

        """

        Splits data, scales features, and trains the model.

        """

        # 1. Split Data (80% Train, 20% Test)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        

        # 2. Scale Data (Normalization)

        # We fit the scaler ONLY on training data to avoid data leakage

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.X_test = self.scaler.transform(self.X_test)

        

        # 3. Train Model

        self.model.fit(X_train_scaled, y_train)

        

        # 4. Internal Prediction for Evaluation

        self.y_pred = self.model.predict(self.X_test)

        

        return accuracy_score(self.y_test, self.y_pred)



    def get_evaluation_metrics(self):

        """

        Returns confusion matrix and classification report.

        """

        cm = confusion_matrix(self.y_test, self.y_pred)

        report = classification_report(self.y_test, self.y_pred)

        return cm, report



    def predict_new_sample(self, input_features):

        """

        Predicts potability for a single water sample.

        Expected Input: List of 9 floats

        """

        # Reshape input to 2D array (1 sample, n features)

        data_array = np.array(input_features).reshape(1, -1)

        

        # Apply the SAME scaling used during training

        scaled_data = self.scaler.transform(data_array)

        

        # Predict class

        prediction = self.model.predict(scaled_data)

        

        # Return readable result

        return "Safe (Potable)" if prediction[0] == 1 else "Unsafe (Not Potable)"