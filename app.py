
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load and prepare data
@st.cache_data
def load_data():
    fake = pd.read_excel("Fake.xlsx")
    true = pd.read_excel("True.xlsx")
    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake[["text", "label"]], true[["text", "label"]]])
    data.dropna(inplace=True)
    return data

# Train and cache the model pipeline
@st.cache_resource
def train_model(data):
    X = data["text"]
    y = data["label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
        ("model", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Streamlit app interface
def main():
    st.title("📰 Fake News Detector")
    st.write("Enter a news article below to check if it is Real or Fake.")

    user_input = st.text_area("📝 Paste news article here:", height=200)

    if st.button("Check News"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some news content.")
        else:
            data = load_data()
            model = train_model(data)
            prediction = model.predict([user_input])[0]
            result = "✅ Real News" if prediction == 1 else "❌ Fake News"
            st.subheader("Result:")
            st.success(result)

if __name__ == "__main__":
    main()
