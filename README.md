# Intelligent-Medical-Chatbot
An Intelligent Medical Chatbot for Symptom Analysis and Assistance 
# Description 

This project is designed to develop a medical chatbot that predicts potential diseases based on user-provided symptoms and recommends suitable medical assistance. By integrating reliable datasets and telemedicine APIs, it bridges that gap between symptoms analysis and healthcare access. 

The project addresses the growing need for accessible healthcare solutions, particularly in scenarios where professional medical assistance is not immediately available. 

# Problem Statement 

This project aims to develop a medical chatbot that allows users to provide or select a list of symptoms and receive possible disease predictions along with recommendations for appropriate medical assistance. Additionally, that chatbot may connect users with telemedicine platforms using APIs to recommend physicians or specialists in the relevant field.  

This project is highly relevant to NLP because it involves understanding and interpreting user-provided natural languages symptom descriptions, matching them with structured medical data and delivering meaningful insights in a conversational manner.

# Dataset Selection 

Dataset(s) that will be used for training and testing our NLP model are still under investigation, due to the need to find and use data that is exhaustive, certified and reliable.  

In particular, we plan to use data coming from the WHO website, where the information is currently available in HTML format and will likely require some pre-processing / web-scraping activity to be downloaded into a usable format; we are also evaluating some Kaggle datasets that are already in csv format but require further analysis to verify the source of that data. 

# WHO Fact Sheets 

Source: https://www.who.int/news-room/fact-sheets 

Non-structured descriptions of diseases which need to be transformed into a structured format for this project. 

# Symptom Disease Mapping Dataset 

Source: 

https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset 

https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning 

Structured data with columns like symptoms, data source might require further checking.

# Expected Outcomes 

**Medical Chatbot**

A user-friendly application incorporating a chatbot that allows users to input symptoms via text and/or select symptoms from a predefined list 

The chatbot integrates with an NLP model to get a list of potential diseases associated with the user-provided symptoms and provide actionable medical recommendations 

**NLP model for symptom-disease mapping 
**
Implementation of a multi-class classification model that maps user provided symptoms to potential diseases 

**Medical Assistance Recommendations 
**
Depending on the list of symptoms and their urgency and gravity, the solution provides recommendations about the most appropriate medical assistance, whether it be urgent care, specialist consultation, or general medical advice. 

**Telemedicine Integrations 
**
To enhance the user experience, the app will integrate with telemedicine platforms through APIs, providing seamless access to recommended physicians and medical experts in the relevant field (e.g., https://developers.mdlivetechnology.com/). This approach ensures timely and accurate healthcare guidance, bridging the gap between symptom detection and professional medical support.

# Evaluation Metrics 

Given the nature of the problem that our solution aims to solve, we plan to consider the following metrics to evaluate the performance of our model: 

**Accuracy,** as we are interested in maximizing proportion of correctly predicted diseases out of all cases 

**Recall,** we want our model to reduce the number of false negatives because we do not want our solution failing to identify a serious condition 

**F1-Score,** which is useful in case of imbalanced datasets and if we want to find a trade-off between recall and precision (reduction of false positives) 
