Autoprice Car Price Prediction

Estimate a car's price with machine learning — through a clean, interactive web app.

Built with Python · Streamlit · Scikit-learn

<br>






</div>


📌 Overview

AutoPrice is a machine-learning web application that estimates a vehicle's price from its technical specifications. Enter details such as engine size, horsepower, dimensions, weight, and fuel economy, then receive a model-generated price estimate through an interactive Streamlit interface.

The project demonstrates an end-to-end beginner-friendly ML workflow: dataset preparation, missing-value handling, model training, evaluation, and an interactive prediction interface.


✨ Features

🚗 Interactive price prediction — enter vehicle specifications in the browser.

🧠 Machine learning — uses a Decision Tree Regressor.

🧹 Data preprocessing — converts numeric fields and handles missing values.

📊 Model metrics — displays Mean Absolute Error (MAE) and R² on a held-out test split.

🎨 Streamlit interface — clean, responsive controls and a dedicated model-information page.

⚡ Simple setup — run locally with a few commands.



🖥️ App Preview

When you run the application, you can:

Open the Price Predictor page.

Enter the car's technical specifications.

Click Predict car price.

View the estimated price.

Open Model Information to review the evaluation metrics and workflow.


🧰 Tech Stack

Technology

Purpose

Python
Core programming language

Streamlit
Interactive web application

Pandas
Dataset loading and data preparation

NumPy
Numerical operations

Scikit-learn
Model training, testing, and evaluation

Decision Tree Regressor


Price prediction algorithm

🗂️ Project Structure

AutoPrice-Car-Price-Prediction/
├── app.py
├── README.md
├── requirements.txt
├── autos_dataset.csv
└── Autoprice car prediction.zip  


⚙️ Getting Started

1. Download or clone the repository

git clone https://github.com/sagarkashid2504-svg/AutoPrice-Car-Price-Prediction.git
cd AutoPrice-Car-Price-Prediction

2. (Recommended) Create a virtual environment

python -m venv .venv
.venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the application

streamlit run app.py

Streamlit will print a local address in the terminal, usually http://localhost:8501. Open that address in your browser.


🔍 How It Works

Automobile Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Train / Test Split
        ↓
Decision Tree Regressor
        ↓
Model Evaluation
        ↓
User Enters Car Specifications
        ↓
Estimated Price Displayed

Load data — read the automobile dataset.

Prepare features — convert required columns to numeric values and replace missing-value markers.

Handle missing data — fill missing feature values using column medians.

Split data — use a training set and a held-out test set.

Train model — fit a Decision Tree Regressor.

Predict — use the submitted vehicle specifications to estimate a price.

Evaluate — display MAE and R² for the test split.


📈 Understanding the Metrics

MAE (Mean Absolute Error): The average absolute difference between test-set prices and model predictions. Lower is generally better, in the same units as the target price.

R² score: Measures how well the model explains price variation on the test set. Values closer to 1 generally indicate a better fit; results depend on the dataset and split.

The displayed values are computed when the app prepares the model. They are not a guarantee of future prediction accuracy.


🚀 Future Enhancements

Save and load a trained model and preprocessing pipeline instead of retraining on each start.

Add make, model, manufacturing year, mileage, fuel type, transmission, and vehicle condition.

Compare Decision Tree, Random Forest, and other regression algorithms.

Add cross-validation and feature-importance visualizations.

Improve mobile layout and deploy the app for public access.



👨‍💻 Author

Sagar Kashid

 Machine Learning Project
