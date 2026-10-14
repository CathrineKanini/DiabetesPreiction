# Diabetes Prediction - Data Science Dashboard
# Built with Streamlit, NumPy and scikit-learn

import html
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION (must be the first Streamlit call)
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction | Data Science",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "trained_model.sav")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.sav")
METRICS_PATH = os.path.join(BASE_DIR, "evaluation_metrics.csv")
CONFUSION_PATH = os.path.join(BASE_DIR, "confusion_matrix.csv")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
<style>
    /* Force a consistent light look so text is always readable,
       even when the viewer's browser or Streamlit is in dark mode. */
    .stApp {
        background-color: #f4f7fb;
    }

    .stApp, .stApp p, .stApp label, .stApp li, .stApp span,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"] {
        color: #102a43;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #102a43;
    }

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    .hero {
        background: linear-gradient(120deg, #102a43, #087f8c);
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff !important;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e4f4f5 !important;
        margin-bottom: 0;
    }

    .result-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #dce5ed;
        margin-top: 20px;
    }

    .result-title {
        color: #102a43;
        font-size: 21px;
        font-weight: 700;
    }

    .muted {
        color: #526777 !important;
        font-size: 14px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background-color: #087f8c;
        color: #ffffff !important;
        border: none;
        border-radius: 9px;
        padding: 0.65rem;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #066570;
        color: #ffffff !important;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }
        .hero {
            padding: 18px;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. LOAD EXISTING MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_artifacts(model_path, scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        fitted_scaler = pickle.load(f)
    return model, fitted_scaler


for path, name in ((MODEL_PATH, "trained_model.sav"), (SCALER_PATH, "scaler.sav")):
    if not os.path.exists(path):
        st.error(
            f"{name} was not found. Check that it is in the same folder "
            "as this script and included in your project repository."
        )
        st.stop()

try:
    loaded_model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
except Exception as error:
    st.error(
        "The saved model or scaler could not be loaded. "
        "This is usually a scikit-learn version mismatch between "
        "training and deployment. Check the logs and pin the same "
        "scikit-learn version in requirements.txt."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# 4. HELPERS
# --------------------------------------------------

def render_html(markup: str) -> None:
    """Render HTML safely.

    Markdown treats lines indented by 4+ spaces as code blocks, which
    prints raw HTML as text. Stripping each line avoids that.
    """
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def show_header(title: str, subtitle: str) -> None:
    render_html(
        f"""
        <div class="hero">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """
    )


def metrics_from_confusion(cm: pd.DataFrame):
    """Derive binary metrics from a confusion matrix (rows = actual)."""
    try:
        cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        tn, fp = cm.loc[0, 0], cm.loc[0, 1]
        fn, tp = cm.loc[1, 0], cm.loc[1, 1]
        total = tn + fp + fn + tp
        if total == 0:
            return None
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return pd.DataFrame(
            {
                "metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "value": [(tp + tn) / total, precision, recall, f1],
            }
        )
    except Exception:
        return None


def show_metrics(metrics_df: pd.DataFrame) -> None:
    cols = st.columns(min(len(metrics_df), 4))
    for i, row in enumerate(metrics_df.itertuples(index=False)):
        with cols[i % len(cols)]:
            st.metric(str(row.metric), f"{row.value:.3f}")
    st.subheader("Metric comparison")
    st.bar_chart(metrics_df.set_index("metric")["value"])


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🩺 Health Analytics")
    st.caption("Machine Learning Project")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Prediction", "Model Performance", "About the Model"],
    )

    st.divider()
    st.caption(f"Model: {type(loaded_model).__name__}")
    st.caption(f"Preprocessing: {type(scaler).__name__}")
    st.divider()
    st.caption("Educational project. Not a medical diagnosis.")


# --------------------------------------------------
# 6. PREDICTION PAGE
# --------------------------------------------------

if page == "Prediction":

    show_header(
        "Diabetes Prediction System",
        "Explore a machine-learning classification using eight "
        "health-related input features.",
    )

    st.subheader("Enter the input measurements")
    st.write(
        "Use the same units and feature order that were used when "
        "the model was trained."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=30, value=0, step=1,
                help="Number of pregnancies.",
            )
            glucose = st.number_input(
                "Glucose", min_value=0.0, max_value=500.0, value=100.0,
                help="Use the units from your training dataset.",
            )
            blood_pressure = st.number_input(
                "Blood Pressure", min_value=0.0, max_value=300.0, value=70.0
            )
            skin_thickness = st.number_input(
                "Skin Thickness", min_value=0.0, max_value=150.0, value=20.0
            )

        with col2:
            insulin = st.number_input(
                "Insulin", min_value=0.0, max_value=3000.0, value=80.0
            )
            bmi = st.number_input(
                "BMI", min_value=0.0, max_value=100.0, value=25.0
            )
            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=10.0, value=0.3,
            )
            age = st.number_input(
                "Age", min_value=1, max_value=120, value=30, step=1
            )

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        # A zero in these fields is usually a missing value in the
        # classic Pima dataset; warn rather than silently predicting.
        zero_fields = [
            name
            for name, value in (
                ("Glucose", glucose),
                ("Blood Pressure", blood_pressure),
                ("BMI", bmi),
            )
            if value == 0
        ]
        if zero_fields:
            st.warning(
                "These values are 0, which is not physiologically "
                f"realistic: {', '.join(zero_fields)}. "
                "The result may be unreliable."
            )

        input_data = np.asarray(
            [[
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, pedigree, age,
            ]],
            dtype=float,
        )

        try:
            scaled_data = scaler.transform(input_data)
            prediction = loaded_model.predict(scaled_data)[0]

            if prediction == 0:
                result_title = "Class 0: Negative prediction"
                result_message = "The model assigned this input to class 0."
            elif prediction == 1:
                result_title = "Class 1: Positive prediction"
                result_message = "The model assigned this input to class 1."
            else:
                result_title = f"Predicted class: {prediction}"
                result_message = "The model returned a class other than 0 or 1."

            render_html(
                f"""
                <div class="result-card">
                    <div class="result-title">{html.escape(str(result_title))}</div>
                    <p class="muted">{html.escape(result_message)}</p>
                </div>
                """
            )

            with st.expander("View prediction details"):
                st.write("Input feature order:")
                st.code(", ".join(FEATURES))

                st.write("Scaled input:")
                st.dataframe(pd.DataFrame(scaled_data, columns=FEATURES))

                if hasattr(loaded_model, "decision_function"):
                    decision = loaded_model.decision_function(scaled_data)
                    st.write("Decision score (distance from the boundary):")
                    st.write(round(float(np.asarray(decision).ravel()[0]), 4))

                if hasattr(loaded_model, "predict_proba"):
                    try:
                        proba = loaded_model.predict_proba(scaled_data)[0]
                        st.write("Class probabilities:")
                        st.dataframe(
                            pd.DataFrame(
                                [proba],
                                columns=[
                                    f"Class {c}" for c in loaded_model.classes_
                                ],
                            )
                        )
                    except Exception:
                        pass  # probability=False at training time

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))

    st.divider()
    st.caption(
        "Important: This experimental model is for educational purposes "
        "only. It cannot diagnose diabetes or replace assessment by a "
        "qualified healthcare professional."
    )


# --------------------------------------------------
# 7. MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    show_header(
        "Model Performance",
        "Review evaluation results from a held-out test dataset.",
    )

    st.info(
        "Performance metrics must come from actual labeled test data. "
        "A prediction from a single input does not measure the accuracy "
        "of the model."
    )

    # Load confusion matrix first so it can also back up the metrics.
    cm_matrix = None
    if os.path.exists(CONFUSION_PATH):
        try:
            cm_df = pd.read_csv(CONFUSION_PATH)
            if {"actual", "predicted", "count"}.issubset(cm_df.columns):
                cm_matrix = cm_df.pivot_table(
                    index="actual",
                    columns="predicted",
                    values="count",
                    aggfunc="sum",
                    fill_value=0,
                )
            else:
                st.error(
                    "confusion_matrix.csv must contain 'actual', "
                    "'predicted', and 'count' columns."
                )
        except Exception as error:
            st.error("Could not read the confusion matrix.")
            st.code(str(error))

    st.subheader("Evaluation metrics")

    metrics_df = None
    source_note = ""

    if os.path.exists(METRICS_PATH):
        try:
            raw = pd.read_csv(METRICS_PATH)
            if {"metric", "value"}.issubset(raw.columns):
                raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
                metrics_df = raw.dropna(subset=["value"])
                source_note = "Source: evaluation_metrics.csv"
            else:
                st.error(
                    "evaluation_metrics.csv must contain 'metric' and "
                    "'value' columns."
                )
        except Exception as error:
            st.error("Could not read evaluation metrics.")
            st.code(str(error))

    if (metrics_df is None or metrics_df.empty) and cm_matrix is not None:
        metrics_df = metrics_from_confusion(cm_matrix)
        source_note = "Source: calculated from confusion_matrix.csv"

    if metrics_df is not None and not metrics_df.empty:
        show_metrics(metrics_df)
        st.caption(source_note)
    elif not os.path.exists(METRICS_PATH):
        st.warning(
            "Evaluation results have not been added yet. Create "
            "evaluation_metrics.csv using metrics calculated from your "
            "labeled test set."
        )
        st.markdown(
            """
Expected file: `evaluation_metrics.csv`

| metric | value |
|---|---:|
| Accuracy | *your value* |
| Precision | *your value* |
| Recall | *your value* |
| F1 Score | *your value* |
"""
        )

    st.divider()
    st.subheader("Confusion matrix")

    if cm_matrix is not None:
        st.dataframe(cm_matrix)
        st.caption("Rows: actual class. Columns: predicted class.")
    elif not os.path.exists(CONFUSION_PATH):
        st.write(
            "Add a confusion_matrix.csv file (columns: actual, predicted, "
            "count) generated from real test labels and model predictions."
        )


# --------------------------------------------------
# 8. ABOUT PAGE
# --------------------------------------------------

elif page == "About the Model":

    show_header(
        "About This Project",
        "A machine-learning application for exploring binary classification.",
    )

    st.subheader("Project overview")
    st.write(
        "This project demonstrates how a trained machine-learning model "
        "can be integrated into a web application. Users enter eight "
        "health-related measurements. The app applies the saved "
        "preprocessing scaler and passes the transformed features to the "
        "saved classifier."
    )

    st.subheader("Technology stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Python")
        st.write("Application programming language.")
    with c2:
        st.markdown("### Streamlit")
        st.write("Interactive web interface.")
    with c3:
        st.markdown("### Scikit-learn")
        st.write("Model and preprocessing tools.")

    st.subheader("Model details")
    details = {
        "Estimator": type(loaded_model).__name__,
        "Preprocessor": type(scaler).__name__,
        "Input features": len(FEATURES),
        "Model classes": str(getattr(loaded_model, "classes_", "Not available")),
    }
    # Convert values to strings so Arrow serialization never fails
    # on a mixed int/str column.
    st.table(
        pd.DataFrame(
            [(k, str(v)) for k, v in details.items()],
            columns=["Property", "Value"],
        )
    )

    st.subheader("Prediction workflow")
    st.markdown(
        """
1. Collect eight input measurements.
2. Convert the inputs to numeric values.
3. Apply the saved scaler.
4. Generate a prediction with the saved classifier.
5. Display the predicted class.
"""
    )

    st.warning(
        "This project is not a clinically validated diagnostic system. "
        "Its output must not be used to make healthcare decisions."
    )
    st.caption(
        "The displayed performance depends on independent evaluation "
        "using appropriately labeled test data."
    )
