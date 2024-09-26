# Digital Healthcare System

## Overview
The **Digital Healthcare System** is designed to predict multiple diseases such as **diabetes**, **heart disease**, and **breast cancer** using advanced machine learning models. It also integrates a chatbot for patient interaction and medical information retrieval. The system aims to assist healthcare professionals in early disease diagnosis, improving patient outcomes, and reducing healthcare costs. Its modular architecture ensures scalability and easy integration of various components.

## Features

### 1. Disease Prediction Modules:
- Supports prediction for:
  - **Diabetes**
  - **Heart Disease**
  - **Breast Cancer**
- Implements machine learning models with high accuracy:
  - **XGBoost**
  - **Random Forest**
  - **Support Vector Machines (SVM)**

### 2. Chatbot Integration:
- Uses **Natural Language Processing (NLP)** techniques to interact with users.
- Provides medical information and guides users through their queries.
- Integrates **Hugging Face Transformers** for context-aware, intelligent responses.

### 3. User-Friendly Interface:
- Developed with **Streamlit** to provide an intuitive and interactive interface.
- Supports both disease prediction and chatbot interaction.

## Tech Stack

- **Backend**: Python (scikit-learn, XGBoost, Hugging Face Transformers)
- **Frontend**: Streamlit
- **Libraries**:
  - **Pandas**, **NumPy**: Data manipulation and numerical computations
  - **Matplotlib**, **Seaborn**: Data visualization
  - **FAISS**, **LangChain**: Embedding and conversational retrieval
- **Machine Learning Models**: XGBoost, Random Forest, SVM
- **Natural Language Processing**: Hugging Face Transformers for chatbot integration

## Datasets

- **Diabetes Prediction**: [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- **Heart Disease Prediction**: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Breast Cancer Prediction**: [Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
  
## Documentation

For detailed documentation, please refer to the [project documentation](https://drive.google.com/file/d/1YPqP9aENgvgRfkPR7FXk4G90DH6kLtWU/view?usp=sharing).


## Installation

To set up the Digital Healthcare System on your local machine, follow these steps:

### 1. Clone the Repository:
```bash
git clone https://github.com/your-repo/digital-healthcare-system.git
cd digital-healthcare-system
```
### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit App:
```bash
streamlit run app.py
```
## Usage
- **Disease Prediction:** Input relevant medical data (e.g., age, gender, BMI) to predict the likelihood of diabetes, heart disease, or breast cancer.
- **Chatbot:** Interact with the chatbot by asking medical-related questions. It will provide context-aware responses and useful medical information.

## Models and Performance
- **Diabetes Prediction:** Achieved accuracy of 97.13%.
- **Heart Disease Prediction:** Achieved accuracy of 92.35%.
- **Breast Cancer Prediction:** Achieved accuracy of 96.46%.

## Future Work
- **Multilingual Support:** Expand chatbot capabilities to support multiple languages.
- **Integration with EHR Systems:** Integrate with Electronic Health Record (EHR) systems to provide real-time data exchange and support telehealth services.
- **Data Expansion:** Continuously update models with real-time data for improved accuracy and personalization.

## License
- This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
- For any questions or further information, please contact:
## Email: ibrahimallabbad9@gmail.com


