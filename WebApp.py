import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
# Load saved models
MODELS_PATH = {
    'Diabetes': 'C:\\Users\\AcTivE\\Desktop\\Digital Healthcare System\\NootBooks\\Diabetes\\Diabetes_model.joblib',
    'Heart Disease': 'C:\\Users\\AcTivE\\Desktop\\Digital Healthcare System\\NootBooks\\Heart Attack\\HeartDisease.joblib',
    'Breast Cancer': 'C:\\Users\\AcTivE\\Desktop\\Digital Healthcare System\\NootBooks\\Breast Cencer\\diagnosis.joblib'
}

MODELS = {disease: joblib.load(open(path, 'rb')) for disease, path in MODELS_PATH.items()}

# Set page title and favicon
st.set_page_config(page_title="Digital Healthcare System", page_icon=":heartbeat:")




def get_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

def display_result(prediction, disease):
    if prediction == 1:
        st.error(f"Risk of {disease} Detected!")
    else:
        st.success(f"No Risk of {disease} Detected.")

def predict_disease(disease, input_df):

    if st.button(f"Predict {disease}"):
        prediction = get_prediction(MODELS[disease], input_df)
        display_result(prediction, disease)

def main():
   # Sidebar for navigation
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',
                               ['Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Breast Cancer Prediction',
                                'ChatBot'],  # Add 'ChatBot' option
                               icons=['activity', 'heart', 'person', 'chat'],  # Add chat icon
                               default_index=0)

    if selected == 'Diabetes Prediction':
        predict_diabetes()
    elif selected == 'Heart Disease Prediction':
        predict_heart_disease()
    elif selected == 'Breast Cancer Prediction':
        predict_breast_cancer()
    elif selected == 'ChatBot':  # Handle ChatBot option
        chat_bot()

def predict_diabetes():
    st.title("Please provide the following information")
    input_df = get_diabetes_input()
    predict_disease('Diabetes', input_df)

def get_diabetes_input():
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=150, value=25, step=1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    smoking_history = st.selectbox("Smoking History", ["Non-smoker", "Past smoker", "Current smoker"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, value=5.0, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, value=100.0, step=1.0)

    input_data = {
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "bmi": [bmi],
        "HbA1c_level": [HbA1c_level],
        "blood_glucose_level": [blood_glucose_level],
        "gender_Male": [1 if gender == "Male" else 0],
        "smoking_history_non-smoker": [1 if smoking_history == "Non-smoker" else 0],
        "smoking_history_past_smoker": [1 if smoking_history == "Past smoker" else 0]
    }
    return pd.DataFrame(input_data)

def predict_heart_disease():
    st.title("Please provide the following information")
    input_df = get_heart_disease_input()
    predict_disease('Heart Disease', input_df)

def get_heart_disease_input():
    age = st.number_input("Age", min_value=0, max_value=150, value=40, step=1)
    sex = st.selectbox("Gender", ["Male", "Female"])
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=1)
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=140, step=1)
    cholesterol = st.number_input("Cholesterol Level", min_value=0.0, value=289.0, step=0.1)
    fasting_bs = st.radio("Fasting Blood Sugar", ["< 120 mg/dl", ">= 120 mg/dl"], index=1)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], index=1)
    max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=300, value=172, step=1)
    exercise_angina = st.radio("Exercise Angina", ["No", "Yes"], index=0)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"], index=2)

    # Convert categorical variables to numerical representations
    chest_pain_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    resting_ecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    st_slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    sex_numeric = 1 if sex == "Male" else 0
    fasting_bs_numeric = 1 if fasting_bs == ">= 120 mg/dl" else 0
    exercise_angina_numeric = 1 if exercise_angina == "Yes" else 0

    # Prepare input data
    input_data = {
        "Age": [age],
        "Sex": [sex_numeric],
        "ChestPainType": [chest_pain_mapping[chest_pain_type]],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs_numeric],
        "RestingECG": [resting_ecg_mapping[resting_ecg]],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina_numeric],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_mapping[st_slope]]
    }

    return pd.DataFrame(input_data)


def predict_breast_cancer():
    st.title("Please provide the following information")
    input_df = get_breast_cancer_input()
    predict_disease('Breast Cancer', input_df)

def get_breast_cancer_input():
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1184, step=0.01)
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.2776, step=0.01)
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2419, step=0.01)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.07871, step=0.01)
    texture_se = st.number_input("Texture SE", min_value=0.0, value=0.9053, step=0.01)
    area_se = st.number_input("Area SE", min_value=0.0, value=153.4, step=0.01)
    smoothness_se = st.number_input("Smoothness SE", min_value=0.0, value=0.006399, step=0.01)
    compactness_se = st.number_input("Compactness SE", min_value=0.0, value=0.04904, step=0.01)
    concavity_se = st.number_input("Concavity SE", min_value=0.0, value=0.05373, step=0.01)
    concave_points_se = st.number_input("Concave Points SE", min_value=0.0, value=0.01587, step=0.01)
    symmetry_se = st.number_input("Symmetry SE", min_value=0.0, value=0.03003, step=0.001)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, value=0.006193, step=0.01)
    texture_worst = st.number_input("Texture Worst", min_value=0.0, value=17.33, step=0.01)
    area_worst = st.number_input("Area Worst", min_value=0.0, value=2019.0, step=0.01)
    smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, value=0.1622, step=0.01)
    compactness_worst = st.number_input("Compactness Worst", min_value=0.0, value=0.6656, step=0.01)
    concavity_worst = st.number_input("Concavity Worst", min_value=0.0, value=0.7119, step=0.01)
    concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, value=0.2654, step=0.01)
    symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, value=0.4601, step=0.01)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, value=0.1189, step=0.01)

    # Prepare input data
    input_data = {
        "smoothness_mean": [smoothness_mean],
        "compactness_mean": [compactness_mean],
        "symmetry_mean": [symmetry_mean],
        "fractal_dimension_mean": [fractal_dimension_mean],
        "texture_se": [texture_se],
        "area_se": [area_se],
        "smoothness_se": [smoothness_se],
        "compactness_se": [compactness_se],
        "concavity_se": [concavity_se],
        "concave_points_se": [concave_points_se],
        "symmetry_se": [symmetry_se],
        "fractal_dimension_se": [fractal_dimension_se],
        "texture_worst": [texture_worst],
        "area_worst": [area_worst],
        "smoothness_worst": [smoothness_worst],
        "compactness_worst": [compactness_worst],
        "concavity_worst": [concavity_worst],
        "concave_points_worst": [concave_points_worst],
        "symmetry_worst": [symmetry_worst],
        "fractal_dimension_worst": [fractal_dimension_worst]
    }

    return pd.DataFrame(input_data)

def chat_bot():
    st.title("HealthCare ChatBot 🧑🏽‍⚕️")
    
    # Load PDF documents, create embeddings, vector store, LLM, and memory 
    loader = DirectoryLoader('C:\\Users\\AcTivE\\Desktop\\Digital Healthcare System\\Data', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                        config={'max_new_tokens': 128, 'temperature': 0.01})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
        
    def conversation_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    def initialize_session_state():
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

    def display_chat_history():
        initialize_session_state()

        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Ask about your Health", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversation_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)  # Append new generated response

        with reply_container:
            # Ensure 'generated' key exists in session state
            if 'generated' in st.session_state:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
            else:
                st.write("No chat history yet.")  # Handle case where 'generated' key is missing

    # Display the chat history
    display_chat_history()



if __name__ == "__main__":
    main()

# Add footer and disclaimer
st.sidebar.markdown('---')
st.sidebar.write("Disclaimer: This system is for educational purposes only. Consult a healthcare professional for medical advice.")
#links to GitHub and LinkedIn
st.sidebar.write("Find me on:") 
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-IbrahimAl--Labbad-blue?style=for-the-badge&logo=github)](https://github.com/IbrahimAl-Labbad)")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ibrahim%20Al--Labbad-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ibrahim-al-labbad-1bbb5327a/)")
