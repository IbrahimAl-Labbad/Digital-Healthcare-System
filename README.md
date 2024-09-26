Digital Healthcare System
Project Overview
This project is a Digital Healthcare System aimed at predicting multiple diseases (such as diabetes, heart disease, and breast cancer) using advanced machine learning models. It also integrates a chatbot for patient interaction and medical information retrieval.

The system is designed to assist healthcare professionals in diagnosing diseases early, thus improving patient outcomes and reducing healthcare costs. The system uses modular architecture to ensure scalability, maintainability, and seamless integration of its diverse components.

Features
Disease Prediction Modules:

Supports the prediction of diabetes, heart disease, and breast cancer.
Implements machine learning models (e.g., XGBoost, SVM, Random Forest) with high accuracy.
Chatbot Integration:

Chatbot leverages NLP techniques to interact with users, providing them with relevant medical information and guiding them through their queries.
Uses advanced language models to ensure interactive, context-aware responses.
User-Friendly Interface:

Developed using Streamlit, providing an intuitive interface for both disease predictions and chatbot interaction.
Tech Stack
Backend: Python (scikit-learn, XGBoost, Hugging Face Transformers)
Frontend: Streamlit
Libraries:
Pandas, NumPy: Data manipulation and numerical computations.
Matplotlib, Seaborn: Data visualization.
FAISS, LangChain: Embedding and conversational retrieval.
Machine Learning: XGBoost, Random Forest, Support Vector Machines (SVM)
Natural Language Processing (NLP): Hugging Face Transformers for chatbot interaction.
Datasets
Diabetes Prediction Dataset:
Link: [Diabetes Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
Heart Disease Prediction Dataset:
Link: [Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
Breast Cancer Prediction Dataset:
Link: [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
Documentation 
Link: https://drive.google.com/file/d/1YPqP9aENgvgRfkPR7FXk4G90DH6kLtWU/view?usp=sharing
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/digital-healthcare-system.git
cd digital-healthcare-system
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Usage
To use the Disease Prediction Modules, input the relevant medical data (e.g., age, gender, BMI) for the system to predict the likelihood of diabetes, heart disease, or breast cancer.
For the Chatbot, ask questions related to medical conditions, and the chatbot will provide relevant information and guidance.

Models and Performance
Diabetes Prediction: Achieved accuracy of 97.13%
Heart Disease Prediction: Achieved accuracy of 92.35%
Breast Cancer Prediction: Achieved accuracy of 96.46%
Future Work
Multilingual Support: Expand chatbot capabilities to support multiple languages.
Integration with EHR Systems: Integrate with Electronic Health Record systems to provide real-time data exchange and support telehealth services.
Data Expansion: Continuously update models with real-time data for improved accuracy and personalization.
License
This project is licensed under the MIT License - see the LICENSE file for details.
