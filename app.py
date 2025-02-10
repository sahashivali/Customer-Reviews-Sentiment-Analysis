import streamlit as st
import joblib

# Load your trained model and vectorizer
model = joblib.load('final_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define the text cleaning function
def cleaning(review_text):
    return review_text.lower()  # Example: converting text to lowercase

# Define the prediction function
def predict_review(review_text):
    processed_text = cleaning(review_text)
    text_count_df = vectorizer.transform([processed_text])
    prediction = model.predict(text_count_df)[0]
    return "Recommended" if prediction == 1 else "Not Recommended"

# Streamlit app layout
st.title("Review Sentiment Prediction")
st.write("Enter a review to predict its sentiment.")

review_input = st.text_area("Review Text")

if st.button("Predict"):
    if review_input:
        result = predict_review(review_input)
        if result == "Recommended":
            st.success(result)  # Green box for recommended
        else:
            st.error(result)  # Red box for not recommended
    else:
        st.warning("Please enter a review.")