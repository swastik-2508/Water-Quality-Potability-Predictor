# AquaSafe: Water Quality Potability Predictor 💧
**Domain**: Environment & SustainabilityTech Stack: Python, Machine Learning (Random Forest)
## 📖 Project Overview
*AquaSafe* is a machine learning-based application designed to analyze water quality metrics and predict whether water is safe for human consumption. By processing physicochemical parameters such as pH, Turbidity, and Hardness, the system provides an automated "Potable" or "Not Potable" classification using a Random Forest Classifier.This project demonstrates the application of modular programming, data preprocessing, and supervised learning to solve real-world environmental challenges.
## ✨ Key Features
**Automated Data Generation:** Automatically creates a synthetic dataset if one is not found.Data Cleaning: Handles missing values using mean imputation and normalizes features using Standard Scaling.Machine Learning: Uses an ensemble Random Forest model for robust prediction accuracy.

**Visualization:** Generates Correlation Heatmaps and Confusion Matrices automatically.Interactive Mode: Allows users to input specific sensor data manually to test water samples.

**Report Generation:** Includes a script to automatically generate a formatted MS Word (.docx) project report.

## 📂 Project Structure
Ensure your folder looks like this to avoid import errors:AquaSafe_Project/
│

├── main.py                  # The main entry point (Run this file)

├── logic.py                 # The bckend ML class and algorithms

├── generate_dataset.py      # Script to create dummy CSV data

├── generate_report_docx.py  # Script to generate the Word report

├── requirements.txt         # List of libraries

│

├── data/                    # Folder for dataset

│   └── water_potability.csv # Generated automatically

│

└── screenshots/             # Folder where graphs are saved

    ├── confusion_matrix.png

    └── correlation_heatmap.png
## 🚀 How to Run
1. Run the Prediction SystemThis will train the model, save the graphs, and let you test custom inputs.python main.py
Follow the on-screen prompts. Type y to enter manual testing mode.
2. Generate the Project ReportThis will create a AquaSafe_Project_Report.docx file in your folder with the Abstract, Methodology, and Conclusion.python generate_report_docx.py

## 📊 MethodologyInput: 
The system accepts 9 parameters: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, and Turbidity.Preprocessing: * Missing values are filled with the column mean.Data is standardized (Mean=0, Variance=1).

Model: A Random Forest Classifier (100 estimators) is trained on 80% of the data.

Output: The model predicts binary class 0 (Unsafe) or 1 (Safe).

## 👨‍💻 Student Details
Name: Swastik kumar Barik 

Reg No:25BOE10064 

Course: Fundamentals in AIML 