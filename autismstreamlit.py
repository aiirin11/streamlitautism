
import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model_path = "autismfyp_model.sav"
scaler_path = "autismscaler.sav"

try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Custom CSS for styling
st.markdown("""
&lt;style&gt;
    /* Main background */
    .stApp {
        background: linear-gradient(to right, #a1c4fd, #c2e9fb);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Form container */
    .stForm {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #001b61;
        color: white;
        border-radius: 30px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #0034a1;
        transform: translateY(-2px);
    }
    
    /* Question cards */
    .question-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.12);
    }
    
    /* Result styling */
    .negative-result {
        background: linear-gradient(135deg, #e0f7fa, #b2ebf2);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 150, 136, 0.3);
        border: 3px solid #4db6ac;
        text-align: center;
    }
    
    .positive-result {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        border: 3px solid #ef9a9a;
        text-align: center;
    }
&lt;/style&gt;
""", unsafe_allow_html=True)

# App state management
if 'stage' not in st.session_state:
    st.session_state.stage = 0
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

def reset_form():
    st.session_state.stage = 0
    st.session_state.form_data = {}

# Page 1: Basic Information
if st.session_state.stage == 0:
    st.title("Autism Spectrum Disorder Screening")
    st.markdown("### Please provide some basic information")
    
    with st.form("basic_info"):
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Select one", "Male", "Female"])
        jaundice = st.selectbox("Had jaundice at birth?", ["Select one", "Yes", "No"])
        relation = st.selectbox("Relationship to person being screened", 
                              ["Select one", "Self", "Family", "Healthcare professional", "Others"])
        
        submitted = st.form_submit_button("Next")
        if submitted:
            if "Select one" in [gender, jaundice, relation]:
                st.error("Please complete all fields")
            else:
                # Convert to model input format
                st.session_state.form_data['age'] = age
                st.session_state.form_data['gender'] = 1 if gender == "Male" else 0
                st.session_state.form_data['jaundice'] = 1 if jaundice == "Yes" else 0
                
                relation_map = {
                    "Self": 3,
                    "Family": 0,
                    "Healthcare professional": 1,
                    "Others": 2
                }
                st.session_state.form_data['relation'] = relation_map[relation]
                
                st.session_state.stage = 1
                st.experimental_rerun()

# Page 2: Questionnaire
elif st.session_state.stage == 1:
    st.title("Autism Screening Questionnaire")
    st.markdown("### Please answer these questions about social interactions")
    
    questions = [
        "S/he often notices small sounds when others do not",
        "S/he usually concentrates more on the whole picture, rather than the small details",
        "In a social group, s/he can easily keep track of several different people's conversations",
        "S/he finds it easy to go back and forth between different activities",
        "S/he doesn't know how to keep a conversation going with his/her peers",
        "S/he is good at social chit-chat",
        "When s/he is read a story, s/he finds it difficult to work out the character's intentions or feelings",
        "When s/he was in preschool, s/he used to enjoy playing games involving pretending with other children",
        "S/he finds it easy to work out what someone is thinking or feeling just by looking at their face",
        "S/he finds it hard to make new friends"
    ]
    
    responses = []
    with st.form("questionnaire"):
        for i, question in enumerate(questions, 1):
            with st.container():
                st.markdown(f'&lt;div class="question-card"&gt;', unsafe_allow_html=True)
                st.markdown(f"**Q{i}:** {question}")
                response = st.radio(
                    f"Response for Q{i}",
                    ["Definitely Agree", "Slightly Agree", "Slightly Disagree", "Definitely Disagree"],
                    key=f"q{i}",
                    horizontal=True
                )
                st.markdown('&lt;/div&gt;', unsafe_allow_html=True)
                
                # Map responses to binary values (1 for agree, 0 for disagree)
                responses.append(1 if "Agree" in response else 0)
        
        submitted = st.form_submit_button("Submit Answers")
        if submitted:
            st.session_state.form_data['responses'] = responses
            st.session_state.stage = 2
            st.experimental_rerun()

# Page 3: Results
elif st.session_state.stage == 2:
    try:
        # Prepare input data
        responses = st.session_state.form_data['responses']
        input_data = responses + [
            st.session_state.form_data['age'],
            st.session_state.form_data['gender'],
            st.session_state.form_data['jaundice'],
            st.session_state.form_data['relation']
        ]
        
        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        
        # Make prediction
        prediction = model.predict(std_data)[0]
        
        # Display results
        if prediction == 1:
            st.markdown('&lt;div class="positive-result"&gt;', unsafe_allow_html=True)
            st.markdown("### RESULT")
            st.markdown("""
            &lt;p style='color: #c62828; font-size: 1.2rem;'&gt;
            THE SYSTEM PREDICTS THAT YOU MAY HAVE&lt;br&gt;
            AUTISM SPECTRUM DISORDER (ASD). PLEASE&lt;br&gt;
            CONSIDER SEEING A MEDICAL SPECIALIST FOR&lt;br&gt;
            FURTHER ASSESSMENT.
            &lt;/p&gt;
            """, unsafe_allow_html=True)
            st.markdown("**TAKE CARE!**")
            st.markdown('&lt;/div&gt;', unsafe_allow_html=True)
        else:
            st.markdown('&lt;div class="negative-result"&gt;', unsafe_allow_html=True)
            st.markdown("### RESULT")
            st.markdown("""
            &lt;p style='color: #00796b; font-size: 1.2rem;'&gt;
            ALL WELL!&lt;br&gt;
            THE SYSTEM FOUND NO AUTISM&lt;br&gt;
            SPECTRUM DISORDER (ASD) TRAITS
            &lt;/p&gt;
            """, unsafe_allow_html=True)
            st.markdown('&lt;/div&gt;', unsafe_allow_html=True)
        
        if st.button("Start New Screening"):
            reset_form()
            st.experimental_rerun()
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        if st.button("Try Again"):
            reset_form()
            st.experimental_rerun()

