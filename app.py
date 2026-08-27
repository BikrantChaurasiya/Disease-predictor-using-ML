import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Disease Prediction Using ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
        color: #ff1493;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: purple;
        margin-bottom: 30px;
    }

    /* Model cards */
    .model-card {
        border: 1px solid #dddddd;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .model-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .prediction {
        font-size: 20px;
        font-weight: 700;
        color: #008000;
    }

    .accuracy {
        font-size: 17px;
        font-weight: 600;
    }

    .confidence {
        font-size: 17px;
        font-weight: 600;
    }

    /* Patient result */
    .patient-result {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    /* Disclaimer */
    .disclaimer {
        margin-top: 30px;
        padding: 18px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #664d03;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 15px;
        color: #777777;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    'DISEASE PREDICTION USING MACHINE LEARNING'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter minimum three symptoms to get prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASETS AUTOMATICALLY
# ============================================================

@st.cache_data
def load_data():
    """
    Automatically load both datasets from the project folder.

    Users do NOT upload anything.
    """

    train = pd.read_csv("training_disease.csv")
    test = pd.read_csv("testing_disease.csv")

    return train, test


try:

    train, test = load_data()

except FileNotFoundError:

    st.error(
        "Dataset files were not found. "
        "Make sure training_disease.csv and testing_disease.csv "
        "are in the same GitHub repository as app.py."
    )

    st.stop()


# ============================================================
# VALIDATE DATASETS
# ============================================================

if "prognosis" not in train.columns:

    st.error(
        "The training dataset must contain a 'prognosis' column."
    )

    st.stop()


if "prognosis" not in test.columns:

    st.error(
        "The testing dataset must contain a 'prognosis' column."
    )

    st.stop()


# ============================================================
# PREPARE TRAINING AND TESTING DATA
# ============================================================

X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]

X_test = test.drop("prognosis", axis=1)
y_test = test["prognosis"]


# ============================================================
# MAKE TEST DATA MATCH TRAINING FEATURES
# ============================================================

# This ensures that the test data has exactly the same
# feature columns and order as the training data.

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# ============================================================
# AUTOMATICALLY GET ALL SYMPTOMS
# ============================================================

all_symptoms = sorted(
    X_train.columns.tolist()
)

symptoms = ["Select Here"] + all_symptoms


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(X_train, y_train):

    # Decision Tree
    decision_tree = DecisionTreeClassifier(
        random_state=42
    )

    # Random Forest
    random_forest = RandomForestClassifier(
        random_state=42
    )

    # Naive Bayes
    naive_bayes = GaussianNB()

    # KNN
    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    # Train models
    decision_tree.fit(
        X_train,
        y_train
    )

    random_forest.fit(
        X_train,
        y_train
    )

    naive_bayes.fit(
        X_train,
        y_train
    )

    knn.fit(
        X_train,
        y_train
    )

    return (
        decision_tree,
        random_forest,
        naive_bayes,
        knn
    )


(
    decision_tree,
    random_forest,
    naive_bayes,
    knn
) = train_models(
    X_train,
    y_train
)


# ============================================================
# CALCULATE MODEL ACCURACY
# ============================================================

dt_test_prediction = decision_tree.predict(X_test)

rf_test_prediction = random_forest.predict(X_test)

nb_test_prediction = naive_bayes.predict(X_test)

knn_test_prediction = knn.predict(X_test)


dt_accuracy = accuracy_score(
    y_test,
    dt_test_prediction
) * 100


rf_accuracy = accuracy_score(
    y_test,
    rf_test_prediction
) * 100


nb_accuracy = accuracy_score(
    y_test,
    nb_test_prediction
) * 100


knn_accuracy = accuracy_score(
    y_test,
    knn_test_prediction
) * 100


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("### 👤 Patient Information")

patient_name = st.text_input(
    "Patient Name",
    placeholder="Enter patient name"
)


st.markdown("### 🩺 Select Symptoms")

col1, col2 = st.columns(2)


with col1:

    symptom1 = st.selectbox(
        "Symptom 1",
        symptoms,
        key="symptom1"
    )

    symptom2 = st.selectbox(
        "Symptom 2",
        symptoms,
        key="symptom2"
    )

    symptom3 = st.selectbox(
        "Symptom 3",
        symptoms,
        key="symptom3"
    )


with col2:

    symptom4 = st.selectbox(
        "Symptom 4",
        symptoms,
        key="symptom4"
    )

    symptom5 = st.selectbox(
        "Symptom 5",
        symptoms,
        key="symptom5"
    )


# ============================================================
# BUTTONS
# ============================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 1, 2]
)


with button_col1:

    predict_button = st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    )


with button_col2:

    clear_button = st.button(
        "🔄 Clear",
        use_container_width=True
    )


# ============================================================
# CLEAR BUTTON
# ============================================================

if clear_button:

    st.session_state.clear()

    st.rerun()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def create_input_vector(selected_symptoms):

    input_vector = np.zeros(
        len(X_train.columns)
    )

    for symptom in selected_symptoms:

        if symptom in X_train.columns:

            symptom_index = X_train.columns.get_loc(
                symptom
            )

            input_vector[symptom_index] = 1

    return input_vector.reshape(1, -1)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Validate patient name
    if not patient_name.strip():

        st.error(
            "⚠️ Please enter the patient name."
        )

        st.stop()


    # Collect symptoms
    selected_symptoms = [
        symptom1,
        symptom2,
        symptom3,
        symptom4,
        symptom5
    ]


    # Remove "Select Here"
    selected_symptoms = [
        symptom
        for symptom in selected_symptoms
        if symptom != "Select Here"
    ]


    # Remove duplicates
    selected_symptoms = list(
        dict.fromkeys(
            selected_symptoms
        )
    )


    # Minimum 3 symptoms
    if len(selected_symptoms) < 3:

        st.error(
            "⚠️ Please select at least 3 different symptoms."
        )

        st.stop()


    # Create model input
    input_data = create_input_vector(
        selected_symptoms
    )


    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    dt_prediction = decision_tree.predict(
        input_data
    )[0]

    rf_prediction = random_forest.predict(
        input_data
    )[0]

    nb_prediction = naive_bayes.predict(
        input_data
    )[0]

    knn_prediction = knn.predict(
        input_data
    )[0]


    # ========================================================
    # MODEL CONFIDENCE
    # ========================================================

    dt_confidence = (
        np.max(
            decision_tree.predict_proba(
                input_data
            )
        ) * 100
    )


    rf_confidence = (
        np.max(
            random_forest.predict_proba(
                input_data
            )
        ) * 100
    )


    nb_confidence = (
        np.max(
            naive_bayes.predict_proba(
                input_data
            )
        ) * 100
    )


    knn_confidence = (
        np.max(
            knn.predict_proba(
                input_data
            )
        ) * 100
    )


    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        f"""
        <div class="patient-result">
        👤 Patient: {patient_name}<br>
        🩺 Selected Symptoms: {", ".join(selected_symptoms)}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # RESULTS TITLE
    # ========================================================

    st.markdown(
        "## 📊 Prediction Results"
    )


    # ========================================================
    # FIRST ROW
    # ========================================================

    result_col1, result_col2 = st.columns(2)


    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    with result_col1:

        st.markdown(
            '<div class="model-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="model-title">'
            '🌳 Decision Tree'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction">'
            f'Prediction: {dt_prediction}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Accuracy:** {dt_accuracy:.2f}%"
        )

        st.write(
            f"**Confidence:** {dt_confidence:.2f}%"
        )

        st.progress(
            int(min(dt_confidence, 100))
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    with result_col2:

        st.markdown(
            '<div class="model-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="model-title">'
            '🌲 Random Forest'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction">'
            f'Prediction: {rf_prediction}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Accuracy:** {rf_accuracy:.2f}%"
        )

        st.write(
            f"**Confidence:** {rf_confidence:.2f}%"
        )

        st.progress(
            int(min(rf_confidence, 100))
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # SECOND ROW
    # ========================================================

    result_col3, result_col4 = st.columns(2)


    # --------------------------------------------------------
    # NAIVE BAYES
    # --------------------------------------------------------

    with result_col3:

        st.markdown(
            '<div class="model-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="model-title">'
            '📊 Naive Bayes'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction">'
            f'Prediction: {nb_prediction}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Accuracy:** {nb_accuracy:.2f}%"
        )

        st.write(
            f"**Confidence:** {nb_confidence:.2f}%"
        )

        st.progress(
            int(min(nb_confidence, 100))
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    with result_col4:

        st.markdown(
            '<div class="model-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="model-title">'
            '👥 KNN'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction">'
            f'Prediction: {knn_prediction}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Accuracy:** {knn_accuracy:.2f}%"
        )

        st.write(
            f"**Confidence:** {knn_confidence:.2f}%"
        )

        st.progress(
            int(min(knn_confidence, 100))
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("ℹ️ About this application"):

    st.write(
        f"**Training samples:** {len(train)}"
    )

    st.write(
        f"**Testing samples:** {len(test)}"
    )

    st.write(
        f"**Available symptoms:** {len(all_symptoms)}"
    )

    st.write(
        f"**Disease classes:** "
        f"{train['prognosis'].nunique()}"
    )

    st.write(
        "**Machine Learning Models:** "
        "Decision Tree, Random Forest, Naive Bayes, KNN"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <b>⚠️ Medical Disclaimer</b><br><br>

    This application is developed for educational and research
    purposes only. The predictions generated by this system should
    not be considered a medical diagnosis and should not replace
    consultation with a qualified healthcare professional.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🩺 Disease Prediction Using Machine Learning
        <br>
        Built with Python, Scikit-learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
