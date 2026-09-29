import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# --------------------------------
# NLTK
# --------------------------------

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# --------------------------------
# Text preprocessing
# SAME as training notebook
# --------------------------------

def transform_text(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Tokenization
    text = nltk.word_tokenize(text)

    # 3. Remove special characters
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # 4. Remove stopwords and punctuation
    for i in text:

        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # 5. Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# --------------------------------
# Load trained model
# --------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# --------------------------------
# Streamlit UI
# --------------------------------

st.title("📧 Email/SMS Spam Detector")

st.write(
    "Enter an email or message and check whether it is Spam or Not Spam."
)

input_sms = st.text_area(
    "Enter your message",
    height=150
)


# --------------------------------
# Prediction
# --------------------------------

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Convert text to TF-IDF
        vector_input = vectorizer.transform(
            [transformed_sms]
        )

        # Prediction
        result = model.predict(vector_input)[0]

        # Probability
        probability = model.predict_proba(vector_input)[0]

        # Display result
        if result == 1:

            st.error("🚨 Spam Message")

            st.write(
                f"Spam probability: {probability[1] * 100:.2f}%"
            )

        else:

            st.success("✅ Not Spam Message")

            st.write(
                f"Not Spam probability: {probability[0] * 100:.2f}%"
            )