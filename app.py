
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ============================================================
# BYTE ME - AI-BASED NETWORK FIREWALL CLASSIFICATION SYSTEM
# Hands-on 3: Upgraded Streamlit Application
# ============================================================

st.set_page_config(
    page_title="Byte Me | AI Network Firewall",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- MODEL CONFIGURATION --------------------

MODEL_FILE = "byteme_genovisa_projecthandson2_model.joblib"

MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILE

FEATURES = [
    "Source Port",
    "Destination Port",
    "NAT Source Port",
    "NAT Destination Port",
    "Bytes",
    "Bytes Sent",
    "Bytes Received",
    "Packets",
    "Elapsed Time (sec)",
    "pkts_sent",
    "pkts_received"
]

PORT_FEATURES = [
    "Source Port",
    "Destination Port",
    "NAT Source Port",
    "NAT Destination Port"
]

CLASS_INFO = {
    "allow": {
        "description": "The traffic is classified as allowed by the machine learning model.",
        "color": "#16A34A"
    },
    "deny": {
        "description": "The traffic is classified as denied.",
        "color": "#DC2626"
    },
    "drop": {
        "description": "The traffic is classified to be dropped.",
        "color": "#EA580C"
    },
    "reset-both": {
        "description": "The traffic is classified for a connection reset.",
        "color": "#7C3AED"
    }
}


# -------------------- CUSTOM CSS --------------------

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #F8FAFC;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #151B26;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }

    /* MAIN TITLES */
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #60A5FA !important;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #CBD5E1 !important;
        font-size: 15px;
        margin-bottom: 25px;
    }

    /* GENERAL TEXT */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp label {
        color: #F8FAFC;
    }

    [data-testid="stWidgetLabel"] p {
        color: #F8FAFC !important;
    }

    /* NUMBER INPUTS */
    [data-testid="stNumberInput"] input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        background-color: #263449 !important;
    }

    /* SELECT BOX - KEEP STREAMLIT DEFAULT MENU BEHAVIOR */
    [data-baseweb="select"] input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-baseweb="select"] > div {
        background-color: #263449 !important;
    }

    /* DROPDOWN OPTIONS */
    [data-baseweb="popover"] {
        background-color: #1E293B !important;
    }

    [role="option"] {
        color: #FFFFFF !important;
        background-color: #1E293B !important;
    }

    /* BUTTONS */
    .stButton button {
        color: #FFFFFF !important;
        background-color: #2563EB;
        border: none;
    }

    /* CARDS */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .result-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .small-note {
        color: #CBD5E1 !important;
        font-size: 13px;
    }

/* EXAMPLE PRESET DROPDOWN TEXT */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label p {
    color: #FFFFFF !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background-color: #263449 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}

div[data-baseweb="popover"] [role="option"] {
    color: #FFFFFF !important;
    background-color: #1E293B !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD SAVED MODEL --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. "
            "Place the model file in the same folder as app.py."
        )

    loaded_model = joblib.load(MODEL_PATH)

    if not hasattr(loaded_model, "predict"):
        raise ValueError("The saved file is not a valid prediction model.")

    if not hasattr(loaded_model, "predict_proba"):
        raise ValueError("The saved model does not support probability scores.")

    if not hasattr(loaded_model, "named_steps"):
        raise ValueError("The saved model is not the expected sklearn Pipeline.")

    if "classifier" not in loaded_model.named_steps:
        raise ValueError("The expected classifier step was not found.")

    if hasattr(loaded_model, "feature_names_in_"):
        saved_features = list(loaded_model.feature_names_in_)

        if saved_features != FEATURES:
            raise ValueError(
                "The model feature names or order do not match the application."
            )

    return loaded_model


try:
    model = load_model()
    model_classes = list(model.classes_)
except Exception as error:
    st.error(f"Unable to load the saved model: {error}")
    st.stop()


# -------------------- DATA VALIDATION --------------------

def validate_traffic_data(data):
    """
    Validate traffic data before classification.

    Port values above 65535 are allowed through validation
    so the application can apply its separate DENY rule.
    """

    missing_features = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        return False, (
            "Missing required features: "
            + ", ".join(missing_features)
        )

    numeric = data[FEATURES].copy()

    for feature in FEATURES:
        numeric[feature] = pd.to_numeric(
            numeric[feature],
            errors="coerce"
        )

    if numeric.isnull().any().any():
        return False, "Please provide valid numeric values for all features."

    values = numeric.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        return False, "Infinite or invalid numeric values are not allowed."

    if (numeric < 0).any().any():
        return False, "Negative values are not allowed."

    if (numeric % 1 != 0).any().any():
        return False, "All input values must be whole numbers."

    if (numeric > np.iinfo(np.int64).max).any().any():
        return False, "One or more values are too large to process."

    return True, numeric.astype("int64")


def get_port_limit_violations(data):
    """Identify port values exceeding the valid port maximum."""

    violations = []

    for feature in PORT_FEATURES:
        if (data[feature] > 65535).any():
            violations.append(feature)

    return violations


# -------------------- CLASSIFICATION FUNCTION --------------------

def classify_data(data):
    """
    Classify valid traffic using the saved ML model.

    If any port exceeds 65535, the application applies
    a separate rule-based DENY decision instead.
    """

    results = data[FEATURES].copy()

    rule_mask = data[PORT_FEATURES].gt(65535).any(axis=1)
    model_mask = ~rule_mask

    predictions = pd.Series(
        "deny",
        index=data.index,
        dtype="object"
    )

    decision_sources = pd.Series(
        "Rule-based Firewall Decision",
        index=data.index,
        dtype="object"
    )

    reasons = pd.Series(
        "Port limit exceeded (maximum valid port: 65535).",
        index=data.index,
        dtype="object"
    )

    scores = pd.Series(
        np.nan,
        index=data.index,
        dtype="float64"
    )

    # Only send valid-port rows to the ML model.
    if model_mask.any():

        valid_data = data.loc[model_mask, FEATURES]

        model_predictions = model.predict(valid_data)
        model_probabilities = model.predict_proba(valid_data)

        predictions.loc[model_mask] = model_predictions

        decision_sources.loc[model_mask] = "Machine Learning Model"

        reasons.loc[model_mask] = [
            CLASS_INFO.get(
                prediction,
                {"description": "Model classification completed."}
            )["description"]
            for prediction in model_predictions
        ]

        scores.loc[model_mask] = (
            model_probabilities.max(axis=1) * 100
        ).round(2)

    results["Predicted Firewall Action"] = predictions
    results["Decision Source"] = decision_sources
    results["Decision Explanation"] = reasons
    results["Top Model Score (%)"] = scores

    return results


# -------------------- SIDEBAR NAVIGATION --------------------

with st.sidebar:
    st.markdown("## 🛡️ BYTE ME")
    st.caption("AI-Based Network Firewall Classification System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Manual Classifier",
            "CSV Traffic Analyzer",
            "About the Model"
        ]
    )

    st.divider()

    st.caption("Machine Learning Project")
    st.caption("IT-4B | Byte Me Group")

    st.caption("Model: Saved Decision Tree Pipeline")


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">🛡️ Byte Me Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Based Network Firewall Classification System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### System Overview")

    st.write(
        "Byte Me is a machine learning application that analyzes "
        "network traffic features and predicts a firewall action."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "Decision Tree")

    with col2:
        st.metric("Output Classes", "4")

    with col3:
        st.metric("Input Features", len(FEATURES))

    st.divider()

    st.markdown("### Firewall Classification Categories")

    category_columns = st.columns(4)

    for column, category in zip(
        category_columns,
        ["allow", "deny", "drop", "reset-both"]
    ):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="color:{CLASS_INFO[category]['color']}">
                        {category.upper()}
                    </h3>
                    <p class="small-note">
                        {CLASS_INFO[category]['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown("### Available Features")

    st.write("**Manual Classifier**")
    st.write("Enter network traffic values and classify one record.")

    st.write("**CSV Traffic Analyzer**")
    st.write("Upload a CSV file to classify multiple traffic records.")

    st.write("**About the Model**")
    st.write("View the model information, features, and limitations.")

    st.info(
        "This application is an academic demonstration. "
        "It predicts firewall actions but does not control or block "
        "real network traffic."
    )


# ============================================================
# PAGE 2: MANUAL CLASSIFIER
# ============================================================

elif page == "Manual Classifier":

    st.markdown(
        '<div class="main-title">🔍 Manual Traffic Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter the network traffic details to get a classification.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Example Presets")

    preset = st.selectbox(
        "Choose an example:",
        [
            "Custom Input",
            "Known Allow Example",
            "Contrasting Traffic Example",
            "Maximum Port Boundary Example",
            "Port Exceeded Example"
        ]
    )

    presets = {
        "Known Allow Example": [
            5156, 443, 0, 0, 200, 100, 100, 2, 1, 1, 1
        ],
        "Contrasting Traffic Example": [
            80, 8080, 5000, 8080, 5000, 3000,
            2000, 50, 30, 30, 20
        ],
        "Maximum Port Boundary Example": [
            5156, 65535, 0, 0, 200, 100, 100,
            2, 1, 1, 1
        ],
        "Port Exceeded Example": [
            5156, 65536, 0, 0, 200, 100, 100,
            2, 1, 1, 1
        ]
    }

    if preset == "Custom Input":
        default_values = [
            5156, 443, 0, 0, 200, 100, 100,
            2, 1, 1, 1
        ]
    else:
        default_values = presets[preset]

    st.divider()

    st.markdown("### Network Traffic Input")

    st.caption(
        "All fields must contain non-negative whole numbers. "
        "Port values above 65535 will receive a rule-based DENY."
    )

    input_data = {}

    left_column, right_column = st.columns(2)

    for index, feature in enumerate(FEATURES):

        target_column = (
            left_column if index < 6 else right_column
        )

        with target_column:
            input_data[feature] = st.number_input(
                feature,
                min_value=0,
                value=int(default_values[index]),
                step=1,
                format="%d",
                key=f"manual_{feature}"
            )

    st.divider()

    if st.button(
        "🔍 Classify Network Traffic",
        type="primary",
        use_container_width=True
    ):

        input_frame = pd.DataFrame([input_data], columns=FEATURES)

        valid, validation_result = validate_traffic_data(input_frame)

        if not valid:
            st.error(validation_result)

        else:
            input_frame = validation_result

            rule_violations = get_port_limit_violations(input_frame)

            try:
                if rule_violations:

                    prediction = "deny"
                    scores = None

                    decision_source = "Rule-based Firewall Decision"

                    reason = (
                        "Port limit exceeded. Maximum valid port number "
                        "is 65535. Exceeded feature(s): "
                        + ", ".join(rule_violations)
                    )

                else:

                    prediction = model.predict(input_frame)[0]

                    scores = model.predict_proba(input_frame)[0]

                    decision_source = "Machine Learning Model"

                    reason = CLASS_INFO[prediction]["description"]

                info = CLASS_INFO.get(
                    prediction,
                    CLASS_INFO["deny"]
                )

                st.markdown("### Classification Result")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h2 style="color:{info['color']}">
                            {prediction.upper()}
                        </h2>
                        <p>{reason}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(f"Decision Source: {decision_source}")

                if rule_violations:
                    st.error(reason)

                else:
                    st.success(
                        "The input passed the port validation rule "
                        "and was evaluated by the saved ML model."
                    )

                    score_data = pd.DataFrame({
                        "Firewall Action": model_classes,
                        "Model Score (%)": scores * 100
                    })

                    score_data = score_data.sort_values(
                        "Model Score (%)",
                        ascending=False
                    )

                    st.markdown("### Model Class Probabilities")

                    fig = px.bar(
                        score_data,
                        x="Firewall Action",
                        y="Model Score (%)",
                        color="Firewall Action",
                        text_auto=".2f",
                        title="Prediction Probability by Class"
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        showlegend=False,
                        yaxis_range=[0, 100]
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.caption(
                        "Model scores are prediction probabilities, "
                        "not guaranteed real-world certainty."
                    )

                with st.expander("View Submitted Input"):
                    st.dataframe(
                        input_frame,
                        use_container_width=True
                    )

            except Exception as error:
                st.error(f"Classification error: {error}")


# ============================================================
# PAGE 3: CSV TRAFFIC ANALYZER
# ============================================================

elif page == "CSV Traffic Analyzer":

    st.markdown(
        '<div class="main-title">📂 CSV Traffic Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload a CSV file to classify multiple network traffic records.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### CSV File Requirements")

    st.write(
        "Your CSV file must contain all 11 required network traffic "
        "features listed below."
    )

    with st.expander("View Required CSV Columns"):
        for feature in FEATURES:
            st.write(f"• {feature}")

    st.info(
        "Port values above 65535 are accepted for validation "
        "and receive a separate rule-based DENY decision. "
        "Missing, negative, non-numeric, and decimal values are rejected."
    )

    uploaded_file = st.file_uploader(
        "Upload your network traffic CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_data = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Data Preview")

            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

            st.caption(
                f"Uploaded records: {len(uploaded_data)}"
            )

            valid, validation_result = validate_traffic_data(
                uploaded_data
            )

            if not valid:
                st.error(validation_result)

            else:

                traffic_data = validation_result

                st.success(
                    "CSV validation successful. "
                    "All required features and values are valid."
                )

                port_violations = get_port_limit_violations(
                    traffic_data
                )

                if port_violations:
                    st.warning(
                        "Some records contain port values above 65535. "
                        "Those records will receive a rule-based DENY "
                        "decision instead of an ML prediction."
                    )

                if st.button(
                    "🚀 Classify All Traffic Records",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        results = classify_data(traffic_data)

                        st.session_state["byte_me_results"] = results

                        st.success(
                            "Classification completed successfully!"
                        )

                    except Exception as error:
                        st.error(f"Classification error: {error}")

                if "byte_me_results" in st.session_state:

                    results = st.session_state["byte_me_results"]

                    if len(results) == len(traffic_data):

                        st.divider()

                        st.markdown("### Classification Results")

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        st.markdown("### Classification Summary")

                        counts = (
                            results["Predicted Firewall Action"]
                            .value_counts()
                            .reindex(model_classes, fill_value=0)
                        )

                        metric_columns = st.columns(4)

                        for column, category in zip(
                            metric_columns,
                            model_classes
                        ):

                            with column:
                                st.metric(
                                    category.upper(),
                                    int(counts[category])
                                )

                        chart_data = (
                            counts.reset_index()
                        )

                        chart_data.columns = [
                            "Firewall Action",
                            "Number of Records"
                        ]

                        fig = px.bar(
                            chart_data,
                            x="Firewall Action",
                            y="Number of Records",
                            color="Firewall Action",
                            text_auto=True,
                            title="Predicted Firewall Actions"
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            showlegend=False
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.markdown("### Filter Results")

                        selected_action = st.selectbox(
                            "Choose a firewall action:",
                            ["All"] + model_classes
                        )

                        if selected_action == "All":
                            filtered_results = results
                        else:
                            filtered_results = results[
                                results["Predicted Firewall Action"]
                                == selected_action
                            ]

                        st.dataframe(
                            filtered_results,
                            use_container_width=True
                        )

                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        filtered_csv = filtered_results.to_csv(
                            index=False
                        ).encode("utf-8")

                        download_col1, download_col2 = st.columns(2)

                        with download_col1:
                            st.download_button(
                                "⬇️ Download All Results",
                                data=csv_data,
                                file_name="byte_me_classification_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        with download_col2:
                            st.download_button(
                                "⬇️ Download Filtered Results",
                                data=filtered_csv,
                                file_name="byte_me_filtered_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

        except Exception as error:
            st.error(f"Unable to read the uploaded CSV file: {error}")


# ============================================================
# PAGE 4: ABOUT THE MODEL
# ============================================================

elif page == "About the Model":

    st.markdown(
        '<div class="main-title">ℹ️ About the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Information about the trained model and its application.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Project Information")

    st.write(
        "**Project Title:** Byte Me – AI-Based Network Firewall "
        "Classification System"
    )

    st.write(
        "**Model:** Decision Tree Classifier"
    )

    st.write(
        "**Model File:** " + MODEL_FILE
    )

    st.write(
        "**Framework:** Streamlit"
    )

    st.write(
        "**Model Loading:** Joblib"
    )

    st.write(
        "**Dataset:** UCI Machine Learning Repository – "
        "Internet Firewall Data"
    )

    st.markdown(
        "[Open the UCI Dataset Repository]"
        "(https://archive.ics.uci.edu/dataset/542/internet+firewall+data)"
    )

    st.divider()

    st.markdown("### Model Configuration")

    classifier = model.named_steps["classifier"]

    st.write(
        f"**Criterion:** {classifier.criterion}"
    )

    st.write(
        f"**Maximum Depth:** {classifier.max_depth}"
    )

    st.write(
        f"**Minimum Samples Split:** {classifier.min_samples_split}"
    )

    st.write(
        f"**Random State:** {classifier.random_state}"
    )

    st.write(
        "**Pipeline:** One fitted classifier step. "
        "No scaler, encoder, or imputer is included in this pipeline."
    )

    st.divider()

    st.markdown("### Input Features")

    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Original source port number.",
            "Original destination port number.",
            "Translated source port number.",
            "Translated destination port number.",
            "Total bytes in the traffic record.",
            "Number of bytes sent.",
            "Number of bytes received.",
            "Total number of packets.",
            "Elapsed connection time in seconds.",
            "Number of packets sent.",
            "Number of packets received."
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Output Classes")

    for category in model_classes:

        info = CLASS_INFO.get(category, {})

        st.markdown(
            f"**{category.upper()}** — "
            f"{info.get('description', 'Model output category.')}"
        )

    st.divider()

    st.markdown("### Additional Port Validation Rule")

    st.write(
        "The application includes an additional rule that checks "
        "the four port-related input features."
    )

    st.write(
        "- Port values from 0 to 65535 are passed to the saved "
        "machine learning model."
    )

    st.write(
        "- Port values above 65535 receive a separate "
        "rule-based DENY decision."
    )

    st.write(
        "- Negative, missing, non-numeric, and non-whole-number "
        "inputs are rejected during validation."
    )

    st.warning(
        "The additional DENY rule is an application validation policy. "
        "It is not a class learned by the Decision Tree model. "
        "The application does not block real network traffic."
    )

    st.divider()

    st.markdown("### Model Limitations")

    st.write(
        "The model's predictions depend on the training dataset "
        "and the quality of the supplied input values."
    )

    st.write(
        "The reset-both category has very few examples in the test set. "
        "Its predictions should therefore be interpreted carefully."
    )

    st.write(
        "A high prediction probability does not guarantee that a "
        "real network connection is safe or malicious."
    )

    st.write(
        "This application is designed for academic demonstration "
        "and learning purposes, not as a production firewall."
    )

    st.divider()

    st.markdown("### Dataset Attribution")

    st.write(
        "Internet Firewall Data. UCI Machine Learning Repository."
    )

    st.markdown(
        "https://archive.ics.uci.edu/dataset/542/internet+firewall+data"
    )

    st.caption(
        "Byte Me | IT-4B | Hands-on 3"
    )
