import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# ---------------- NLTK SAFE DOWNLOAD ----------------
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------------- TEXT TRANSFORMATION ----------------
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---------------- LOAD MODEL ----------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------- STREAMLIT UI ----------------
st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)

        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. predict
        result = model.predict(vector_input)[0]

        # 4. output
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
