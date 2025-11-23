import pandas as pd
import numpy as np
import os

def create_dummy_data(num_rows=1000):
    """
    Generates a synthetic dataset similar to the Kaggle Water Quality dataset.
    """
    np.random.seed(42)
    
    # 1. Generate features based on realistic distributions
    data = {
        'pH': np.random.normal(loc=7.0, scale=1.5, size=num_rows),
        'Hardness': np.random.normal(loc=196, scale=30, size=num_rows),
        'Solids': np.random.normal(loc=22000, scale=8000, size=num_rows),
        'Chloramines': np.random.normal(loc=7.1, scale=1.5, size=num_rows),
        'Sulfate': np.random.normal(loc=333, scale=40, size=num_rows),
        'Conductivity': np.random.normal(loc=426, scale=80, size=num_rows),
        'Organic_carbon': np.random.normal(loc=14.2, scale=3.0, size=num_rows),
        'Trihalomethanes': np.random.normal(loc=66, scale=16, size=num_rows),
        'Turbidity': np.random.normal(loc=3.96, scale=0.7, size=num_rows),
        'Potability': np.random.randint(0, 2, size=num_rows)  # 0 or 1
    }
    
    # 2. Add some "logic" so the ML model has something to learn
    # (e.g., if pH is very acidic, water is definitely not potable)
    df = pd.DataFrame(data)
    
    # Logic: Make extreme pH values unsafe
    df.loc[(df['pH'] < 6.0) | (df['pH'] > 9.0), 'Potability'] = 0
    
    # Logic: Make high turbidity unsafe
    df.loc[df['Turbidity'] > 5.0, 'Potability'] = 0
    
    # Logic: Make high Sulfate unsafe
    df.loc[df['Sulfate'] > 450, 'Potability'] = 0
    
    # 3. Clip values to ensure they are positive (Physics check)
    numerical_cols = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                      'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    for col in numerical_cols:
        df[col] = df[col].clip(lower=0.1)
        if col == 'pH':
            df[col] = df[col].clip(upper=14.0)

    # 4. Introduce some missing values (NaN) to test your cleaning logic
    # Set 5% of pH and Sulfate to NaN
    mask_ph = np.random.choice([True, False], size=num_rows, p=[0.05, 0.95])
    df.loc[mask_ph, 'pH'] = np.nan
    
    mask_sulf = np.random.choice([True, False], size=num_rows, p=[0.05, 0.95])
    df.loc[mask_sulf, 'Sulfate'] = np.nan

    return df

if _name_ == "_main_":
    # Ensure directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    df = create_dummy_data(1000)
    
    file_path = 'data/water_potability.csv'
    df.to_csv(file_path, index=False)
    
    print(f"✅ Success! Dataset generated at: {file_path}")
    print("You can now run main.py")