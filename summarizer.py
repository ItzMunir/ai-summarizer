# summarizer.py
from transformers import pipeline

def summarize_text(text, max_length=130, min_length=30):
    """Summarize text into ~3 sentences."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


if __name__ == "__main__":
    sample_text = """
    Artificial Intelligence is transforming the world at a rapid pace. 
    It powers everything from chatbots to self-driving cars, 
    and has potential applications in almost every industry.
    However, it also raises questions about ethics, jobs, and the future of work.
    """
    print("Original text:\n", sample_text)
    print("\nSummary:\n", summarize_text(sample_text))
