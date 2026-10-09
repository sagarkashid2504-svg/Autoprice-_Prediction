import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="AutoPrice | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: Pink; margin-bottom: 1.3rem;
    }
    .hero h1 {color: Pink; margin-bottom: .3rem;}
    .hero p {color: #e8f4f8; margin-bottom: 0;}
    div[data-testid="stMetric"] {
        background: Pink; padding: 1rem; border-radius: 12px;
        border: 1px solid #e5eaf0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🚘 AutoPrice</h1>
      <p>Estimate a car's price using a machine-learning model trained on the automobile dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DATASET_NAME = "autos_dataset.csv"
FEATURES = [
    "symboling", "wheel-base", "length", "width", "height", "curb-weight",
    "num-of-cylinders", "engine-size", "compression-ratio", "horsepower",
    "peak-rpm", "city-mpg", "highway-mpg",
]
CYLINDER_MAP = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "eight": 8, "twelve": 12,
}


def load_dataset():
    """Load CSV from the app folder, the extracted project folder, or the uploaded ZIP."""
    candidates = [
        Path(__file__).resolve().parent / DATASET_NAME,
        Path(__file__).resolve().parent / "Autoprice car prediction" / DATASET_NAME,
        Path.cwd() / DATASET_NAME,
        Path.cwd() / "Autoprice car prediction" / DATASET_NAME,
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    zip_candidates = [
        Path(__file__).resolve().parent / "Autoprice car prediction.zip",
        Path.cwd() / "Autoprice car prediction.zip",
    ]
    for zip_path in zip_candidates:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                matches = [n for n in archive.namelist() if n.endswith("/" + DATASET_NAME)]
                if matches:
                    return pd.read_csv(io.BytesIO(archive.read(matches[0])))

    raise FileNotFoundError(
        "Could not find autos_dataset.csv. Put it beside app.py, extract the project ZIP, "
        "or keep 'Autoprice car prediction.zip' beside app.py."
    )


@st.cache_resource
def prepare_model():
    df = load_dataset().copy()
    if "price" not in df.columns:
        raise ValueError("The dataset must contain a 'price' column.")

    # Match the preprocessing used in the supplied notebook.
    df.replace("?", np.nan, inplace=True)
    df["num-of-cylinders"] = df["num-of-cylinders"].replace(CYLINDER_MAP)

    for col in FEATURES + ["price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price"])
    X = df[FEATURES].copy()
    y = df["price"].astype(float)

    # Fill missing feature values using medians calculated from the dataset.
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    score = r2_score(y_test, predictions)

    return model, medians, X, mae, score, len(df)


try:
    model, medians, training_data, test_mae, test_r2, rows_used = prepare_model()
except Exception as exc:
    st.error(f"Unable to start the prediction app: {exc}")
    st.info(
        "Check that autos_dataset.csv is in the same folder as app.py, or keep "
        "'Autoprice car prediction.zip' beside app.py."
    )
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("About AutoPrice")
st.sidebar.write(
    "Enter the car's technical specifications. The model estimates a price based on "
    "patterns learned from the supplied automobile dataset."
)
st.sidebar.caption("Educational project — predictions are estimates, not official valuations.")
page = st.sidebar.radio("Navigate", ["Price Predictor", "Model Information"])

if page == "Price Predictor":
    st.subheader("Enter car specifications")
    st.write("Adjust the values below, then select **Predict car price**.")

    ranges = {
        "symboling": (-3, 3, 0, 1),
        "wheel-base": (80.0, 125.0, 98.0, 0.1),
        "length": (140.0, 210.0, 175.0, 0.1),
        "width": (60.0, 75.0, 66.0, 0.1),
        "height": (47.0, 65.0, 54.0, 0.1),
        "curb-weight": (1400, 4100, 2600, 10),
        "num-of-cylinders": (2, 12, 4, 1),
        "engine-size": (60, 330, 130, 1),
        "compression-ratio": (7.0, 24.0, 9.0, 0.1),
        "horsepower": (45, 300, 110, 1),
        "peak-rpm": (4000, 7000, 5200, 100),
        "city-mpg": (10, 50, 25, 1),
        "highway-mpg": (15, 55, 30, 1),
    }

    labels = {
        "symboling": "Insurance risk rating (symboling)",
        "wheel-base": "Wheelbase",
        "length": "Car length",
        "width": "Car width",
        "height": "Car height",
        "curb-weight": "Curb weight",
        "num-of-cylinders": "Number of cylinders",
        "engine-size": "Engine size",
        "compression-ratio": "Compression ratio",
        "horsepower": "Horsepower",
        "peak-rpm": "Peak RPM",
        "city-mpg": "City fuel economy (MPG)",
        "highway-mpg": "Highway fuel economy (MPG)",
    }

    left, right = st.columns(2)
    values = {}
    for i, feature in enumerate(FEATURES):
        low, high, default, step = ranges[feature]
        container = left if i % 2 == 0 else right
        with container:
            values[feature] = st.number_input(
                labels[feature],
                min_value=low,
                max_value=high,
                value=default,
                step=step,
                key=f"input_{feature}",
            )

    st.caption("Dimensions and weight use the dataset's original units; MPG means miles per gallon.")
    if st.button("🔍 Predict car price", type="primary", use_container_width=True):
        input_df = pd.DataFrame([values], columns=FEATURES)
        prediction = float(model.predict(input_df)[0])

        st.success("Prediction completed")
        st.metric("Estimated car price", f"${prediction:,.0f}")
        st.caption(
            "The dataset uses prices in its original currency/units. This result is a model estimate "
            "and may differ from a real market price."
        )

elif page == "Model Information":
    st.subheader("Model overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records used", f"{rows_used}")
    c2.metric("Test Mean Absolute Error", f"${test_mae:,.0f}")
    c3.metric("Test R² score", f"{test_r2:.3f}")

    st.markdown("### How it works")
    st.markdown(
        """
        1. Loads the automobile dataset.
        2. Converts missing-value markers and numeric columns into usable values.
        3. Fills missing feature values with the dataset median.
        4. Splits the data into training and testing sets (80% / 20%).
        5. Trains a Decision Tree Regressor and uses it to estimate the price.
        """
    )
    st.warning(
        "This app trains the model when it starts. Results depend on the small supplied dataset "
        "and are intended for learning/demo purposes."
    )

st.divider()
st.caption("AutoPrice • Built with Streamlit and scikit-learn")
