import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import scipy.sparse

# Load dataset
dataset = pd.read_csv("NLP-Chatbot\Dataset\DATASET RESUME.csv")

# Handle NaN values and convert float values to strings
dataset.fillna('', inplace=True)
X_skills = dataset['Skills & Details']
X_tools = dataset['Tools & Technologies']
X_education = dataset['Education Details']
y = dataset["Category"]

# Split dataset into train and test sets
X_train_skills, X_test_skills, y_train, y_test = train_test_split(X_skills, y, test_size=0.2, random_state=42)
X_train_tools, X_test_tools = train_test_split(X_tools, test_size=0.2, random_state=42)
X_train_education, X_test_education = train_test_split(X_education, test_size=0.2, random_state=42)

# Feature extraction (TF-IDF) with limited vocabulary size
max_features_skills = 500  
max_features_tools = 500   
max_features_education = 500 

vectorizer_skills = TfidfVectorizer(max_features=max_features_skills)
vectorizer_tools = TfidfVectorizer(max_features=max_features_tools)
vectorizer_education = TfidfVectorizer(max_features=max_features_education)

X_train_vec_skills = vectorizer_skills.fit_transform(X_train_skills)
X_train_vec_tools = vectorizer_tools.fit_transform(X_train_tools)
X_train_vec_education = vectorizer_education.fit_transform(X_train_education)

X_test_vec_skills = vectorizer_skills.transform(X_test_skills)
X_test_vec_tools = vectorizer_tools.transform(X_test_tools)
X_test_vec_education = vectorizer_education.transform(X_test_education)

# Concatenate feature vectors
X_train_vec = scipy.sparse.hstack([X_train_vec_skills, X_train_vec_tools, X_train_vec_education])
X_test_vec = scipy.sparse.hstack([X_test_vec_skills, X_test_vec_tools, X_test_vec_education])

# Save vectorizers
skills_path = r"NlP-Chatbot\\pkl\\tfidf_vectorizer_skills.pkl"
joblib.dump(vectorizer_skills, skills_path)

tools_path = r"NlP-Chatbot\\pkl\\tfidf_vectorizer_tools.pkl"
joblib.dump(vectorizer_tools, tools_path)

edu_path = r"NlP-Chatbot\\pkl\\tfidf_vectorizer_education.pkl"
joblib.dump(vectorizer_education, edu_path)

# Model training (SVM classifier)
classifier = SVC(kernel='linear')
classifier.fit(X_train_vec, y_train)

# Model evaluation
y_pred = classifier.predict(X_test_vec)

# Save the trained model to disk
classifier_path = r"NlP-Chatbot\\pkl\\nlp_model.pkl"
joblib.dump(classifier, classifier_path)

