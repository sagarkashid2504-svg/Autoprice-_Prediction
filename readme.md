# AutoPrice — Car Price Prediction

## Short Description
AutoPrice is a machine-learning web application built with Streamlit that estimates a car's price from its technical specifications. It uses a Decision Tree Regressor trained on an automobile dataset and provides an interactive interface for entering vehicle details and viewing a predicted price.

## Features
- Predicts estimated car prices from vehicle specifications.
- Interactive Streamlit interface.
- Uses a Decision Tree Regression model.
- Handles missing values in the dataset.
- Displays model evaluation metrics, including Mean Absolute Error (MAE) and R².
- Includes a model-information page.

## Tech Stack
- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit

## Project Structure
```text
Autoprice/
├── app.py
├── autos_dataset.csv
├── train_model.py             # Optional, if you create a separate training script
├── car_price_model.pkl        # Optional, if you save a trained model
└── requirements.txt
```

The current `app.py` version trains the model when the app starts and can read `autos_dataset.csv` from the project folder. If you are using the ZIP-loading option in `app.py`, keep `Autoprice car prediction.zip` beside `app.py` instead.

## Installation and Setup

### 1. Clone or download the project
Download the project files and open the project folder in VS Code or a terminal.

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn
```

You can also create a `requirements.txt` file containing:
```text
streamlit
pandas
numpy
scikit-learn
```

Then install with:
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Place `autos_dataset.csv` in the same folder as `app.py`, or keep the project ZIP beside `app.py` if your app is configured to load the CSV from the ZIP.

The dataset must contain a `price` target column and the feature columns expected by the app.

### 4. Run the app
From the project folder, run:
```bash
streamlit run app.py
```

Streamlit will show a local address in the terminal, usually `http://localhost:8501`. Open that address in your browser.

## How It Works
1. Loads the automobile dataset.
2. Replaces missing-value markers and converts required columns to numeric values.
3. Fills missing feature values using column medians.
4. Splits the data into training and test sets.
5. Trains a Decision Tree Regressor.
6. Accepts vehicle specifications through the Streamlit interface.
7. Returns an estimated price and displays evaluation metrics.

## Model Evaluation
The app reports:
- **MAE (Mean Absolute Error):** the average absolute difference between actual and predicted prices on the test set.
- **R² score:** indicates how well the model explains price variation on the test set. Scores depend on the dataset and split.

These metrics are estimates from the test split and do not guarantee accuracy on real-world market prices.

## Limitations
- Predictions depend on the dataset's size, quality, and price range.
- The dataset may not represent current market prices, local taxes, mileage, vehicle condition, or regional differences.
- The model is intended for educational and demonstration purposes, not as an official vehicle valuation.

## Future Improvements
- Save and load the trained model and preprocessing pipeline with a trusted `.pkl` file.
- Add vehicle make, model, year, mileage, fuel type, and transmission.
- Use cross-validation and compare multiple regression algorithms.
- Improve responsive design and deploy the app online.

## Author
Add your name and project details here.

## License
Add a license if you intend to distribute this project publicly.
