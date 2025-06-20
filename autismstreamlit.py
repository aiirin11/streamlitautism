import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open("autismfyp_model.sav", 'rb'))
scaler = pickle.load(open("autismscaler.sav", 'rb'))

st.set_page_config(page_title="Autism Prediction", layout="centered")

# Session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "form1"

def go_to_form2():
    st.session_state.page = "form2"

def go_to_result():
    st.session_state.page = "result"

# -------- PAGE 1: Basic Info -------- #
if st.session_state.page == "form1":
    st.markdown("""
        <h2 style='text-align:center; color:#001b61;'>Enter Your Details</h2>
    """, unsafe_allow_html=True)

    with st.form("user_info_form"):
        age = st.number_input("Age", min_value=1, step=1)

        gender = st.selectbox("Choose your gender", ["Select", "Male", "Female"])
        jaundice = st.selectbox("Have jaundice during birth?", ["Select", "Yes", "No"])
        relation = st.selectbox("What is your relationship?", [
            "Select", "Family", "Healthcare professional", "Others", "Self"
        ])

        submitted = st.form_submit_button("Next")
        if submitted:
            if "Select" in [gender, jaundice, relation]:
                st.warning("Please complete all selections.")
            else:
                st.session_state.age = age
                st.session_state.gender = 1 if gender == "Male" else 0
                st.session_state.jaundice = 1 if jaundice == "Yes" else 0
                relation_map = {
                    "Family": 0, "Healthcare professional": 1,
                    "Others": 2, "Self": 3
                }
                st.session_state.relation = relation_map[relation]
                go_to_form2()

# -------- PAGE 2: Questionnaire -------- #
elif st.session_state.page == "form2":
    st.markdown("""
        <h2 style='text-align:center; color:#001b61;'>Please tick one option per question only:</h2>
    """, unsafe_allow_html=True)

    with st.form("questionnaire_form"):
        responses = []
        for i in range(1, 11):
            question = f"Q{i}:"
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{i}.**")
            with col2:
                answer = st.radio(
                    label="",
                    options=[
                        "Definitely Agree", "Slightly Agree",
                        "Slightly Disagree", "Definitely Disagree"
                    ],
                    key=f"q{i}",
                    horizontal=True,
                    index=None
                )
                if answer in ["Definitely Agree", "Slightly Agree"]:
                    responses.append(1)
                else:
                    responses.append(0)

        submitted2 = st.form_submit_button("Submit")
        if submitted2:
            st.session_state.responses = responses
            go_to_result()

# -------- PAGE 3: Result -------- #
elif st.session_state.page == "result":
    try:
        features = st.session_state.responses + [
            st.session_state.age,
            st.session_state.gender,
            st.session_state.jaundice,
            st.session_state.relation
        ]
        input_array = np.asarray(features).reshape(1, -1)
        std_data = scaler.transform(input_array)
        prediction = model.predict(std_data)[0]

        if prediction == 1:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffebee, #ffcdd2); padding: 40px 30px;
                            border-radius: 15px; box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3); text-align: center;">
                    <h1 style="color:#c62828;">RESULT</h1>
                    <p style="font-size:20px; color:#d32f2f; font-weight:500;">THE SYSTEM PREDICTS THAT YOU MAY HAVE</p>
                    <p style="font-size:20px; color:#d32f2f; font-weight:500;">AUTISM SPECTRUM DISORDER (ASD).</p>
                    <p style="font-size:20px; color:#d32f2f; font-weight:500;">PLEASE CONSIDER SEEING A MEDICAL SPECIALIST.</p>
                    <p style="font-size:18px; color:#b71c1c; font-weight:bold;">TAKE CARE!</p>
                    <a href="/" onclick="window.location.reload();" style="display:inline-block; margin-top:20px; background-color:#c62828; color:white;
                        padding:12px 30px; border-radius:5px; text-decoration:none;">Start New Screening</a>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #e0f7fa, #b2ebf2); padding: 40px 30px;
                            border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 150, 136, 0.3); text-align: center;">
                    <h1 style="color:#00695c;">RESULT</h1>
                    <p style="font-size:20px; color:#00796b; font-weight:500;">ALL WELL!</p>
                    <p style="font-size:20px; color:#00796b; font-weight:500;">NO AUTISM SPECTRUM DISORDER (ASD) TRAITS FOUND.</p>
                    <a href="/" onclick="window.location.reload();" style="display:inline-block; margin-top:20px; background-color:#00897b; color:white;
                        padding:12px 30px; border-radius:5px; text-decoration:none;">Start New Screening</a>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
