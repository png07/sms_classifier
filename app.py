import streamlit as st
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

ps=PorterStemmer()

def transform_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    y=[]
    y=re.sub('[^a-zA-Z]',' ',text)
    y=y.lower()
    y=y.split()
    y=[ps.stem(word) for word in y if not word in stopwords.words('english')]
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
