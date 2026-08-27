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
    page_title="Disease Prediction",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    color: #ff1493;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: purple;
    font-weight: bold;
    margin-bottom: 25px;
}

.result-box {
    background-color: #d8ffd8;
    padding: 12px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
}

.disclaimer {
    margin-top: 30px;
    padding: 15px;
    border-radius: 8px;
    background-color: #fff3cd;
    border: 1px solid #ffeeba;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">DISEASE PREDICTION USING MACHINE LEARNING</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter minimum three symptoms to get prediction</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    train = pd.read_csv("training_disease.csv")
    test = pd.read_csv("testing_disease.csv")

    return train, test


train, test = load_data()


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]

X_test = test.drop("prognosis", axis=1)
y_test = test["prognosis"]


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(X_train, y_train):

    dt = DecisionTreeClassifier(random_state=42)

    rf = RandomForestClassifier(random_state=42)

    nb = GaussianNB()

    knn = KNeighborsClassifier(n_neighbors=5)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    return dt, rf, nb, knn


dt, rf, nb, knn = train_models(X_train, y_train)


# ============================================================
# MODEL ACCURACY
# ============================================================

dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
nb_pred = nb.predict(X_test)
knn_pred = knn.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred) * 100
rf_accuracy = accuracy_score(y_test, rf_pred) * 100
nb_accuracy = accuracy_score(y_test, nb_pred) * 100
knn_accuracy = accuracy_score(y_test, knn_pred) * 100


# ============================================================
# SYMPTOMS
# ============================================================

symptoms = [
    "Select Here",
    "itching",
    "skin_rash",
    "continuous_sneezing",
    "shivering",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "vomiting",
    "fatigue",
    "weight_loss",
    "cough",
    "high_fever",
    "headache",
    "back_pain",
    "constipation",
    "diarrhoea",
    "nausea",
    "chest_pain",
    "neck_pain",
    "dizziness",
    "loss_of_balance",
    "loss_of_smell"
]


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(2)


with left:

    name = st.text_input(
        "Patient Name",
        placeholder="Enter Patient Name"
    )

    symptom1 = st.selectbox("Symptom 1", symptoms)

    symptom2 = st.selectbox("Symptom 2", symptoms)

    symptom3 = st.selectbox("Symptom 3", symptoms)

    symptom4 = st.selectbox("Symptom 4", symptoms)

    symptom5 = st.selectbox("Symptom 5", symptoms)

    col1, col2 = st.columns(2)

    with col1:
        predict_button = st.button(
            "Predict Disease",
            type="primary",
            use_container_width=True
        )

    with col2:
        clear_button = st.button(
            "Clear",
            use_container_width=True
        )


# ============================================================
# CLEAR
# ============================================================

if clear_button:
    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

results = None

if predict_button:

    if name.strip() == "":
        st.error("Enter Patient Name")

    else:

        selected = [
            s for s in [
                symptom1,
                symptom2,
                symptom3,
                symptom4,
                symptom5
            ]
            if s != "Select Here"
        ]

        # Remove duplicate symptoms
        selected = list(dict.fromkeys(selected))

        if len(selected) < 3:

            st.error("Select minimum 3 symptoms")

        else:

            input_vector = np.zeros(len(X_train.columns))

            for symptom in selected:

                if symptom in X_train.columns:

                    index = X_train.columns.get_loc(symptom)

                    input_vector[index] = 1

            input_vector = input_vector.reshape(1, -1)


            # Predictions

            dt_prediction = dt.predict(input_vector)[0]

            rf_prediction = rf.predict(input_vector)[0]

            nb_prediction = nb.predict(input_vector)[0]

            knn_prediction = knn.predict(input_vector)[0]


            # Confidence

            dt_conf = np.max(
                dt.predict_proba(input_vector)
            ) * 100

            rf_conf = np.max(
                rf.predict_proba(input_vector)
            ) * 100

            nb_conf = np.max(
                nb.predict_proba(input_vector)
            ) * 100

            knn_conf = np.max(
                knn.predict_proba(input_vector)
            ) * 100


            results = {
                "dt": (
                    dt_prediction,
                    dt_accuracy,
                    dt_conf
                ),

                "rf": (
                    rf_prediction,
                    rf_accuracy,
                    rf_conf
                ),

                "nb": (
                    nb_prediction,
                    nb_accuracy,
                    nb_conf
                ),

                "knn": (
                    knn_prediction,
                    knn_accuracy,
                    knn_conf
                )
            }


# ============================================================
# RESULTS
# ============================================================

if results is not None:

    st.markdown("---")

    st.success(
        f"Prediction generated for patient: **{name}**"
    )

    col1, col2 = st.columns(2)

    # -----------------------------
    # Decision Tree
    # -----------------------------

    with col1:

        st.markdown(
            '<div class="section-title">🌳 Decision Tree</div>',
            unsafe_allow_html=True
        )

        prediction, accuracy, confidence = results["dt"]

        st.text_input(
            "Prediction",
            prediction,
            disabled=True,
            key="dt_prediction"
        )

        st.text_input(
            "Accuracy",
            f"{accuracy:.2f}%",
            disabled=True,
            key="dt_accuracy"
        )

        st.text_input(
            "Confidence",
            f"{confidence:.2f}%",
            disabled=True,
            key="dt_confidence"
        )


    # -----------------------------
    # Random Forest
    # -----------------------------

    with col2:

        st.markdown(
            '<div class="section-title">🌲 Random Forest</div>',
            unsafe_allow_html=True
        )

        prediction, accuracy, confidence = results["rf"]

        st.text_input(
            "Prediction",
            prediction,
            disabled=True,
            key="rf_prediction"
        )

        st.text_input(
            "Accuracy",
            f"{accuracy:.2f}%",
            disabled=True,
            key="rf_accuracy"
        )

        st.text_input(
            "Confidence",
            f"{confidence:.2f}%",
            disabled=True,
            key="rf_confidence"
        )


    st.markdown("---")


    col1, col2 = st.columns(2)

    # -----------------------------
    # Naive Bayes
    # -----------------------------

    with col1:

        st.markdown(
            '<div class="section-title">📊 Naive Bayes</div>',
            unsafe_allow_html=True
        )

        prediction, accuracy, confidence = results["nb"]

        st.text_input(
            "Prediction",
            prediction,
            disabled=True,
            key="nb_prediction"
        )

        st.text_input(
            "Accuracy",
            f"{accuracy:.2f}%",
            disabled=True,
            key="nb_accuracy"
        )

        st.text_input(
            "Confidence",
            f"{confidence:.2f}%",
            disabled=True,
            key="nb_confidence"
        )


    # -----------------------------
    # KNN
    # -----------------------------

    with col2:

        st.markdown(
            '<div class="section-title">👥 KNN</div>',
            unsafe_allow_html=True
        )

        prediction, accuracy, confidence = results["knn"]

        st.text_input(
            "Prediction",
            prediction,
            disabled=True,
            key="knn_prediction"
        )

        st.text_input(
            "Accuracy",
            f"{accuracy:.2f}%",
            disabled=True,
            key="knn_accuracy"
        )

        st.text_input(
            "Confidence",
            f"{confidence:.2f}%",
            disabled=True,
            key="knn_confidence"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="disclaimer">

<b>⚠️ Disclaimer:</b>

This application is developed for educational and research purposes only.
The predictions generated by this system should not be considered a medical
diagnosis and should not replace advice from a qualified healthcare professional.

</div>
""", unsafe_allow_html=True)
