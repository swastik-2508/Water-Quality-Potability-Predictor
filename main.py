import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from logic import WaterQualityModel
import generator  # importing your dataset generator script

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'data'
DATA_FILE = 'water_potability.csv'
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
SCREENSHOT_DIR = 'screenshots'

# Ensure directories exist
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

def check_and_prepare_data():
    """
    Checks if the dataset exists. If not, generates it using generate_dataset.py.
    """
    if not os.path.exists(DATA_PATH):
        print(f"[SETUP] '{DATA_PATH}' not found.")
        print("[SETUP] Calling dataset generator...")
        
        # Ensure data folder exists
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            
        # Generate data using the imported module
        df = generator.create_dummy_data(1000) # Generating 1000 rows
        df.to_csv(DATA_PATH, index=False)
        print(f"[SETUP] ✅ Dataset generated and saved to {DATA_PATH}")
    else:
        print(f"[SETUP] Found existing dataset at {DATA_PATH}")

def run_system():
    # 1. Prepare Data
    check_and_prepare_data()

    # 2. Initialize Logic
    print("[INIT] Initializing AquaSafe System...")
    system = WaterQualityModel()

    # 3. Load Data
    print(f"[LOAD] Processing data...")
    X, y, full_df = system.load_and_clean(DATA_PATH)
    
    if X is not None:
        # 4. Train Model
        print("[TRAIN] Training Random Forest Model (this may take a second)...")
        acc = system.train(X, y)
        print(f"   >>> Model Accuracy: {acc*100:.2f}%")

        # 5. Evaluate
        cm, report = system.get_evaluation_metrics()
        print("\n--- Model Performance Report ---")
        print(report)

        # 6. Visualization
        print("[VIZ] Generating reports in 'screenshots' folder...")
        
        # Confusion Matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"{SCREENSHOT_DIR}/confusion_matrix.png")
        plt.close()

        # Correlation Heatmap
        plt.figure(figsize=(10,8))
        sns.heatmap(full_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation")
        plt.tight_layout()
        plt.savefig(f"{SCREENSHOT_DIR}/correlation_heatmap.png")
        plt.close()

        # 7. Manual Input Loop
        while True:
            print("\n" + "="*40)
            print("       MANUAL WATER TESTING MODE")
            print("="*40)
            choice = input("Do you want to predict a new water sample? (y/n): ").strip().lower()
            
            if choice != 'y':
                break
            
            print("\nPlease enter the following parameters:")
            try:
                # We need exactly 9 inputs to match the model's training data
                ph = float(input("   1. pH (0-14): "))
                hardness = float(input("   2. Hardness (mg/L): "))
                solids = float(input("   3. Solids/TDS (ppm): "))
                chloramines = float(input("   4. Chloramines (ppm): "))
                sulfate = float(input("   5. Sulfate (mg/L): "))
                conductivity = float(input("   6. Conductivity (μS/cm): "))
                organic_carbon = float(input("   7. Organic Carbon (ppm): "))
                trihalomethanes = float(input("   8. Trihalomethanes (μg/L): "))
                turbidity = float(input("   9. Turbidity (NTU): "))
                
                user_sample = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
                
                print(f"\n[ANALYZING] Processing data...")
                result = system.predict_new_sample(user_sample)
                print(f"   >>> RESULT: {result}")
                
            except ValueError:
                print("\n[ERROR] Invalid input. Please enter numeric values only.")
        
        print("\n[SUCCESS] Execution complete.")

if _name_ == "_main_":
    run_system()