import streamlit as st
from transformers import pipeline

# Load summarizer model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.title("📝 AI Summarizer")
st.write("Paste an article or text, and I'll shrink it into a snackable summary.")

# User input
article_text = st.text_area("Enter your text here:", height=200)

if st.button("Summarize"):
    if article_text.strip():
        summary = summarizer(article_text, max_length=130, min_length=30, do_sample=False)
        st.subheader("Summary")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("Please paste some text first!")
